import os
import math
import json
import pytz
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

## TODO: Adjust these global constants:
T = 100 #
TR = 1 #???
N_SAMPLE = 128
N_STOCK = 3
COEF_ = 1e10
S_OUTSTANDING = torch.tensor([1.15, 0.32, 0.23]) *1e10 / COEF_
GAMMA = 1/(1/ (8.91*1e-13) + 1/ (4.45 * 1e-12) ) / COEF_
BM_COV = torch.eye(3) #[[1, 0.5], [0.5, 1]]
## END HERE ##

TIMESTAMPS = np.linspace(0, TR, T + 1)
#BM_COV = torch.tensor(BM_COV)
assert len(BM_COV.shape) == 2 and BM_COV.shape[0] == BM_COV.shape[1] and BM_COV.shape[0]
N_BM = BM_COV.shape[0]

## TODO: Adjust this function to get constant processes
## Compute constants processes using dW
def get_constants(dW_std):
    W_std = torch.cumsum(torch.cat((torch.zeros((N_SAMPLE, 1, N_BM)), dW_std), dim=1), dim=1)
    
    mu_tm = torch.tensor([[2.99, 3.71, 3.55]]).repeat(T,1)
    sigma_big = torch.tensor([[72.00, 71.49, 54.80],[71.49, 85.42, 65.86],[54.80, 65.86, 56.84]])
    sigma_md = solve_sigma_md_theoretical(sigma_big) #torch.ones((T, N_STOCK, N_BM)) #???
    sigma_tmd = torch.ones((T, N_STOCK, N_BM)) * sigma_md
    s_tm = torch.ones((T, N_STOCK))
    xi_dd = torch.tensor([[ -2.07, 1.91, 0.64],[1.91, -1.77, -0.59],[0.64 ,-0.59 ,-0.20]]) *1e9 / COEF_
    lam_mm = torch.diag(torch.tensor([0.1269, 0.3354, 0.8595])) * 1e-8 * COEF_ #torch.ones((N_STOCK, N_STOCK))
    alpha_md = torch.ones((N_STOCK, N_BM)) #???
    beta_m = torch.ones(N_STOCK) #???

    return W_std.to(device = DEVICE), mu_tm.to(device = DEVICE), sigma_tmd.to(device = DEVICE), s_tm.to(device = DEVICE), xi_dd.to(device = DEVICE), lam_mm.to(device = DEVICE), alpha_md.to(device = DEVICE), beta_m.to(device = DEVICE)

## Solve for sigma assuming square matrix
def solve_sigma_md_theoretical(sigma_mm_cov):
    evals, evecs = torch.eig(sigma_mm_cov, eigenvectors = True)
    return torch.matmul(evecs, torch.matmul(torch.diag(evals[:,0] ** 0.5), torch.inverse(evecs)))

## Solve for sigma using numerical methods
def solve_sigma_md(sigma_mm_cov, epoch = 1000, lr = 1e-2):
    sigma_md = torch.normal(0, 1, size = (N_STOCK, N_BM))
    sigma_md.requires_grad = True
    for _ in tqdm(range(epoch)):
        target = sigma_md @ sigma_md.T
        loss = torch.sum((sigma_mm_cov - target) ** 2)
        loss.backward()
        sigma_md.data = sigma_md.data - lr * sigma_md.grad
        sigma_md.grad.detach_()
        sigma_md.grad.zero_()
    return torch.abs(sigma_md.data)

## Check if CUDA is avaialble
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
    DEVICE = "cpu"
else:
    print('CUDA is available!  Training on GPU ...')
    DEVICE = "cuda"

## Set seed globally
torch.manual_seed(0)

MULTI_NORMAL = MultivariateNormal(torch.zeros((N_SAMPLE, T, N_BM)), BM_COV)
dW_STD = MULTI_NORMAL.sample().to(device = DEVICE)
W_STD, MU_TM, SIGMA_TMD, S_TM, XI_DD, LAM_MM, ALPHA_MD, BETA_M = get_constants(dW_STD)

## Neural network learning a single value
class S_0(nn.Module):
    def __init__(self, output_dim = 1):
        super(S_0, self).__init__()
        self.s_0 = nn.Linear(1, output_dim)
#        torch.nn.init.constant_(self.s_0.weight, S_INITIAL)
  
    def forward(self, x):
        return self.s_0(x)
        
## Feedforward neural network
class Net(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM_LST, OUTPUT_DIM=1):
        super(Net, self).__init__()
        self.layer_lst = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.layer_lst.append(nn.Linear(INPUT_DIM, HIDDEN_DIM_LST[0]))
        self.bn.append(nn.BatchNorm1d(HIDDEN_DIM_LST[0],momentum=0.1))
        for i in range(1, len(HIDDEN_DIM_LST)):
            self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[i - 1], HIDDEN_DIM_LST[i]))
            self.bn.append(nn.BatchNorm1d(HIDDEN_DIM_LST[i],momentum=0.1))
        self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[-1], OUTPUT_DIM))

    def forward(self, x):
        for i in range(len(self.layer_lst) - 1):
            x = self.layer_lst[i](x)
            x = self.bn[i](x)
            x = F.relu(x)
        return self.layer_lst[-1](x)

## Model wrapper
class ModelFull(nn.Module):
    def __init__(self, predefined_model, is_discretized = False):
        super(ModelFull, self).__init__()
        self.model = predefined_model
        self.is_discretized = is_discretized
    
    def forward(self, tup):
        t, x = tup
        if self.is_discretized:
            return self.model[t](x)
        else:
            return self.model(x)

## Construct arbitrary neural network models with optimizer and scheduler
class ModelFactory:
    def __init__(self, algo, model_name, input_dim, hidden_lst, output_dim, lr, decay, scheduler_step, solver = "Adam", retrain = False):
        assert solver in ["Adam", "SGD", "RMSprop"]
        assert model_name in ["discretized_feedforward", "rnn"]
        assert algo in ["deep_hedging", "fbsde"]
        self.lr = lr
        self.decay = decay
        self.scheduler_step = scheduler_step
        self.solver = solver
        self.model_name = model_name
        self.input_dim = input_dim
        self.hidden_lst = hidden_lst
        self.output_dim = output_dim
        self.model = None
        self.prev_ts = None
        self.algo = algo
        
        if not retrain:
            self.model, self.prev_ts = self.load_latest()
        if self.model is None:
            if model_name == "discretized_feedforward":
                self.model = self.discretized_feedforward()
                if algo == "fbsde":
                    self.model.append(S_0())
                self.model = ModelFull(self.model, is_discretized = True)
            else:
                self.model = self.rnn()
                if algo == "fbsde":
                    model_lst = nn.ModuleList()
                    model_lst.append(self.model)
                    model_lst.append(S_0(self.output_dim))
                    self.model = model_lst
                    self.model = ModelFull(self.model, is_discretized = True)
                else:
                    self.model = ModelFull(self.model, is_discretized = False)
            self.model = self.model.to(device = DEVICE)

    ## TODO: Implement it -- Zhanhao Zhang
    def discretized_feedforward(self):
        model_list = nn.ModuleList()
        for _ in range(len(TIMESTAMPS) - 1):
            model = Net(self.input_dim, self.hidden_lst, self.output_dim)
            model_list.append(model)
        return model_list
    
    ## TODO: Implement it -- Zhanhao Zhang
    def rnn(self):
        return None
    
    def update_model(self, model):
        self.model = model
    
    def prepare_model(self):
        if self.solver == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        elif self.solver == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        else:
            optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.scheduler_step, gamma = self.decay)
        return self.model, optimizer, scheduler, self.prev_ts
    
    def save_to_file(self, curr_ts = None):
        if curr_ts is None:
            curr_ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
        model_save = self.model.cpu()
        torch.save(model_save, f"Models/{self.algo}_{self.model_name}__{curr_ts}.pt")
        return curr_ts
    
    def load_latest(self):
        ts_lst = [f.strip(".pt").split("__")[1] for f in os.listdir("Models/") if f.endswith(".pt") and f.startswith(self.model_name)]
        ts_lst = sorted(ts_lst, reverse=True)
        if len(ts_lst) == 0:
            return None, None
        ts = ts_lst[0]
        model = torch.load(f"Models/{self.algo}_{self.model_name}__{ts}.pt")
        model = model.to(device = DEVICE)
        return model, ts

## TODO: Implement it
## Return tensors of phi_dot and phi at each timestamp t
class DynamicsFactory():
    def __init__(self, ts_lst, dW_std):
        assert ts_lst[0] == 0
        self.dW_std = dW_std
        self.W_std, self.mu_tm, self.sigma_tmd, self.s_tm, self.xi_dd, self.lam_mm, self.alpha_md, self.beta_m = get_constants(dW_std)
        self.lam_mm_half = self.mat_frac_pow(self.lam_mm, 1/2)
        self.lam_mm_negHalf = self.mat_frac_pow(self.lam_mm, -1/2)
        self.alpha_mm_sq = self.alpha_md @ self.alpha_md.T
        self.const_mm = (GAMMA ** (1/2)) * self.lam_mm_negHalf @ self.mat_frac_pow(self.lam_mm_negHalf @ self.alpha_mm_sq @ self.lam_mm_negHalf, 1/2) @ self.lam_mm_half
        self.sigma_tmm_sq = torch.einsum("ijk, ilk -> ijl", self.sigma_tmd, self.sigma_tmd)
        self.sigma_tmm_sq_inv = torch.zeros((T, N_STOCK, N_STOCK))
        #for t in range(T):
        #    self.sigma_tmm_sq_inv[t,:,:] = torch.inverse(self.sigma_tmm_sq[t,:,:]) #self.sigma_tmm_sq_inv[t,:,:] #
        self.xi_std_w = torch.einsum("ijk, kl -> ijl", self.W_std[:,1:,:], self.xi_dd)
        self.phi_stm_bar = 1 / GAMMA * torch.einsum("ijk, ik -> ij", self.sigma_tmm_sq_inv, self.mu_tm) - torch.einsum("jlk, ijk -> ijl", torch.einsum("ijk, ikl -> ijl", self.sigma_tmm_sq_inv, self.sigma_tmd), self.xi_std_w)
    
    def get_constant_processes(self):
        return self.W_std, self.mu_tm, self.sigma_tmd, self.s_tm, self.xi_dd, self.lam_mm, self.alpha_md, self.beta_m
    
    def mat_frac_pow(self, mat, power):
        evals, evecs = torch.eig(mat, eigenvectors = True)
        mat_ret = torch.matmul(evecs, torch.matmul(torch.diag(evals[:,0] ** power), torch.inverse(evecs)))
        return mat_ret
    
    ## TODO: Implement it -- Daran Xu
    def fbsde_quad(self, model):
        # need (1 + T) models in total
        # model output should be
        Z_stmd = torch.zeros((N_SAMPLE, T, N_STOCK, N_BM)).to(device=DEVICE)
        phi_stm = torch.zeros((N_SAMPLE, T + 1, N_STOCK)).to(device=DEVICE)
        phi_stm[:, 0, :] = S_OUTSTANDING / 2
        phi_dot_stm = torch.zeros((N_SAMPLE, T + 1, N_STOCK)).to(device=DEVICE)  # note here phi_dot has T+1 timesteps
        curr_t = torch.ones((N_SAMPLE, 1))
        phi_dot_stm[:, 0, :] = model((-1, curr_t)) #curr_t as dummy input
        for t in range(T):
            phi_stm[:, t + 1, :] = phi_stm[:, t, :] + phi_dot_stm[:, t, :] * TR / T
            x = torch.cat((self.W_std[:, t, :], t / T * TR * curr_t), dim=1).to(device=DEVICE)
            Z_stmd[:, t, :, :] = model((t, x)).reshape(N_SAMPLE, N_STOCK, N_BM)
            phi_dot_stm[:, t + 1, :] = phi_dot_stm[:, t, :] + \
                                       +TR / T * GAMMA * torch.einsum("ij,bj -> bi", torch.mm(
                torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.sigma_tmd[t, :, :].T),
                                                                      (phi_stm[:, t, :] - self.phi_stm_bar[:, t, :]))+\
                +torch.einsum('bik, bk -> bi', Z_stmd[:, t, :, :], self.dW_std[:, t, :])
            """
            +TR / T * GAMMA * torch.einsum("ij,bj -> bi", torch.mm(torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.sigma_tmd[t, :, :].T), phi_stm[:, t, :]) - TR / T * torch.matmul(torch.inverse(self.lam_mm), self.mu_tm[t, :]) + TR / T * GAMMA * torch.einsum("md,sd -> sm", torch.mm(torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.xi_dd), self.W_std[:, t, :]) +\
            +TR/T*GAMMA*torch.einsum("ij,bj -> bi", torch.mm(torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.sigma_tmd[t, :, :].T), (phi_stm[:, t, :]-self.phi_stm_bar[:, t, :]))
            """
            """
            # ??? N
            phi_dot_stm[:, t + 1, :] = phi_dot_stm[:, t, :] + \
                +TR / T * GAMMA * torch.einsum( "ij,bj -> bi" , torch.mm( torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.sigma_tmd[t, :, :].T)  ,  phi_stm[:, t, :]) + \
                +TR / T * GAMMA * torch.mm(torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.xi_dd)[:,0]+ \
                -TR / T * torch.mm(torch.mm(torch.mm(torch.inverse(self.lam_mm), self.sigma_tmd[t, :, :]), self.sigma_tmd[t, :, :].T),S_OUTSTANDING_M1 )[:,0] + \
                +torch.einsum( 'bik, bk -> bi' , Z_stmd[:, t, :, :],self.dW_std[:,t,:])
            # ??? N

            stock_stm[:, t+1 , :] = stock_stm[:, t , :] +\
                +TR / T * GAMMA * torch.mm(torch.mm( self.sigma_tmd[t, :, :], self.sigma_tmd[t, :, :].T),S_OUTSTANDING_M1 )[:,0] +\
                + torch.einsum( 'ij, bj -> bi' , self.sigma_tmd[t, :, :],self.dW_std[:,t,:])"""
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- TBD
    def fbsde_power(self, model):
        pass
    
    ## TODO: Implement it -- Zhanhao Zhang
    def leading_order_quad(self, model = None):
        phi_stm = torch.zeros((N_SAMPLE, T + 1, N_STOCK)).to(device = DEVICE)
        phi_stm[:,0,:] = S_OUTSTANDING / 2
        phi_dot_stm = torch.zeros((N_SAMPLE, T, N_STOCK)).to(device = DEVICE)
        for t in range(T):
            phi_dot_stm[:,t,:] = -(phi_stm[:,t,:] - self.phi_stm_bar[:,t,:]) @ self.const_mm
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * TR / T
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- TBD
    def leading_order_power(self, model = None):
        pass
    
    ## TODO: Implement it -- Zhanhao Zhang
    def ground_truth(self, model = None):
        phi_stm = torch.zeros((N_SAMPLE, T + 1, N_STOCK)).to(device = DEVICE)
        phi_stm[:,0,:] = S_OUTSTANDING / 2
        phi_dot_stm = torch.zeros((N_SAMPLE, T, N_STOCK)).to(device = DEVICE)
        for t in range(T):
            phi_dot_stm[:,t,:] = -(phi_stm[:,t,:] - self.phi_stm_bar[:,t,:]) @ self.const_mm @ torch.tanh(self.const_mm * (T - t))
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * TR / T
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- Zhanhao Zhang
    def deep_hedging(self, model):
        phi_stm = torch.zeros((N_SAMPLE, T + 1, N_STOCK)).to(device = DEVICE)
        phi_stm[:,0,:] = S_OUTSTANDING / 2
        phi_dot_stm = torch.zeros((N_SAMPLE, T, N_STOCK)).to(device = DEVICE)
        curr_t = torch.ones((N_SAMPLE, 1))
        for t in range(T):
            x = torch.cat((self.W_std[:,t,:], curr_t), dim = 1).to(device = DEVICE)
            phi_dot_stm[:,t,:] = model((t, x))
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * TR / T
        return phi_dot_stm, phi_stm

## TODO: Implement it
## Return the loss as a tensor
class LossFactory():
    def __init__(self, ts_lst, dW_std):
        assert ts_lst[0] == 0
        self.dW_std = dW_std
        self.W_std, self.mu_tm, self.sigma_tmd, self.s_tm, self.xi_dd, self.lam_mm, self.alpha_md, self.beta_m = get_constants(dW_std)
        self.mse_loss_func=torch.nn.MSELoss()

    ## TODO: Implement it -- Zhanhao Zhang
    def utility_loss(self, phi_dot_stm, phi_stm, power):
        loss_mat = torch.einsum("ijk, jk -> ij", phi_stm[:,1:,:], self.mu_tm) - GAMMA / 2 * (torch.einsum("ijk, jkl -> ij", phi_stm[:,1:,:], self.sigma_tmd) + torch.einsum("ijk, kl -> ij", self.W_std[:,1:,:], self.xi_dd)) ** 2 - 1 / 2 * torch.einsum("ijk, lk, ijl -> ij", phi_dot_stm, self.lam_mm, phi_dot_stm)
        loss_compact = -torch.sum(loss_mat * TR / T) / N_SAMPLE
        return loss_compact
    
    ## TODO: Implement it -- Zhanhao Zhang
    def fbsde_loss(self, phi_dot_stm, phi_stm):
        Value=phi_dot_stm[:,-1,:]
        Target=torch.zeros_like(Value,device=DEVICE)
        return self.mse_loss_func(Value,Target)


## Write training logs to file
def write_logs(ts_lst, train_args):
    with open("Logs.tsv", "a") as f:
        for i in range(1, len(ts_lst)):
            line = f"{ts_lst[i - 1]}\t{ts_lst[i]}\t{json.dumps(train_args)}\n"
            f.write(line)

## Visualize loss function through training
def visualize_loss(loss_arr, ts):
    plt.plot(loss_arr)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.savefig(f"Plots/loss_{ts}.png")
    plt.close()

## Visualize dynamics of an arbitrary component
def Visualize_dyn(timestamps, arr, ts, name):
    assert name in ["phi", "phi_dot", "sigma", "mu", "s"]
    if name == "phi":
        title = "${\\varphi}_t$"
    elif name == "phi_dot":
        title = "$\dot{\\varphi}_t$"
    elif name == "sigma":
        title = "$\sigma_t$"
    elif name == "mu":
        title = "$\mu_t$"
    else:
        title = "$S_t$"
    arr = arr.cpu().detach().numpy()
    if len(arr.shape) == 2 and arr.shape[1] == 1:
        arr = arr.reshape((-1,))
    if len(arr.shape) == 1:
        plt.plot(timestamps, arr)
    else:
        for i in range(arr.shape[1]):
            plt.plot(timestamps, arr[:, i], label = f"Stock {i + 1}")
    plt.xlabel("T")
    plt.ylabel(title)
    plt.title(title)
    plt.legend()
    plt.savefig(f"Plots/{name}_{ts}.png")
    plt.close()

## The training pipeline
def training_pipeline(algo = "deep_hedging", cost = "quadratic", model_name = "discretized_feedforward", solver = "Adam", hidden_lst = [50], lr = 1e-2, epoch = 1000, decay = 0.1, scheduler_step = 10000, retrain = False):
    assert algo in ["deep_hedging", "fbsde"]
    assert cost in ["quadratic", "power"]
    assert model_name in ["discretized_feedforward", "rnn"]
    assert solver in ["Adam", "SGD", "RMSprop"]
    
    if cost == "quadratic":
        power = 2
    else:
        power = 3 / 2
    
    ## TODO: Change the input dimension accordingly
    if algo == "deep_hedging":
        output_dim = N_STOCK
    else:
        output_dim = N_BM * N_STOCK
    model_factory = ModelFactory(algo, model_name, N_BM + 1, hidden_lst, output_dim, lr, decay, scheduler_step, solver, retrain)
    model, optimizer, scheduler, prev_ts = model_factory.prepare_model()
    
    loss_arr = []
    
    for itr in tqdm(range(epoch)):
        optimizer.zero_grad()
        ## TODO: Implement it
        dW_std = MULTI_NORMAL.sample().to(device = DEVICE)
        dynamic_factory = DynamicsFactory(TIMESTAMPS, dW_std)
        loss_factory = LossFactory(TIMESTAMPS, dW_std)
        
        if algo == "deep_hedging":
            phi_dot_stm, phi_stm = dynamic_factory.deep_hedging(model)
            loss = loss_factory.utility_loss(phi_dot_stm, phi_stm, power)
        else:
            if cost == "quadratic":
                phi_dot_stm, phi_stm = dynamic_factory.fbsde_quad(model)
            else:
                phi_dot_stm, phi_stm = dynamic_factory.fbsde_power(model)
            loss = loss_factory.fbsde_loss(phi_dot_stm, phi_stm)
        ## End here ##
        loss_arr.append(float(loss.data))
        loss.backward()
        
        if torch.isnan(loss.data):
            break
        optimizer.step()
        scheduler.step()
    
    ## Update and save model
    model_factory.update_model(model)
    curr_ts = model_factory.save_to_file()
    
    ## Visualize loss and results
    visualize_loss(loss_arr, curr_ts)
    return model, loss_arr, prev_ts, curr_ts

def evaluation(dW_std, curr_ts, model = None, algo = "deep_hedging", cost = "quadratic", visualize_obs = 0):
    assert algo in ["deep_hedging", "fbsde", "leading_order", "ground_truth"]
    assert cost in ["quadratic", "power"]
    if cost == "quadratic":
        power = 2
    else:
        power = 3 / 2
    dynamic_factory = DynamicsFactory(TIMESTAMPS, dW_std)
    W_std, mu_tm, sigma_tm, s_tm, xi_dd, lam_mm, alpha_md, beta_m = dynamic_factory.get_constant_processes()
    if algo == "deep_hedging":
        phi_dot_stm, phi_stm = dynamic_factory.deep_hedging(model)
    elif algo == "fbsde":
        if cost == "quadratic":
            phi_dot_stm, phi_stm = dynamic_factory.fbsde_quad(model)
        else:
            phi_dot_stm, phi_stm = dynamic_factory.fbsde_power(model)
        ### to match the dim
        phi_dot_stm = phi_dot_stm[:,:-1,:]
    elif algo == "leading_order":
        if cost == "quadratic":
            phi_dot_stm, phi_stm = dynamic_factory.leading_order_quad(model)
        else:
            phi_dot_stm, phi_stm = dynamic_factory.leading_order_power(model)
    else:
        assert cost == "quadratic"
        phi_dot_stm, phi_stm = dynamic_factory.ground_truth(model)
    loss_factory = LossFactory(TIMESTAMPS, dW_std)
    loss = loss_factory.utility_loss(phi_dot_stm, phi_stm, power)
    
    Visualize_dyn(TIMESTAMPS[1:], phi_stm[0,1:,:], curr_ts, "phi")
    Visualize_dyn(TIMESTAMPS[1:], phi_dot_stm[0,:,:], curr_ts, "phi_dot")
    Visualize_dyn(TIMESTAMPS[1:], sigma_tm, curr_ts, "sigma")
    Visualize_dyn(TIMESTAMPS[1:], mu_tm, curr_ts, "mu")
    Visualize_dyn(TIMESTAMPS[1:], s_tm, curr_ts, "s")
    return float(loss.data)
        
## TODO: Adjust the arguments for training
train_args = {
    "algo": "deep_hedging",
    "cost": "quadratic",
    "model_name": "discretized_feedforward",
    "solver": "Adam",
    "hidden_lst": [50, 50, 50],
    "lr": 1e-2,
    "epoch": 1000,
    "decay": 0.1,
    "scheduler_step": 10000,
    "retrain": True,
}

model, loss_arr, prev_ts, curr_ts = training_pipeline(**train_args)
loss_eval = evaluation(dW_STD, curr_ts, model, algo = train_args["algo"], cost = train_args["cost"], visualize_obs = 0)
#loss_eval = evaluation(dW_STD, "test", None, algo = "leading_order", cost = train_args["cost"], visualize_obs = 0)
write_logs([prev_ts, curr_ts], train_args)
