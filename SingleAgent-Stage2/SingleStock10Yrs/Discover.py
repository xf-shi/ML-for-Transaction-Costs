drive_dir = "." #"drive/MyDrive/CFRM/RL/SingleAgent-Stage2"

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
T = 252#1008#10080#5040 #2520 #5000 #
TR = 252#2520 #???
N_SAMPLE = 3000#3000#128
POWER = 3/2 #3/2
N_STOCK = 1 # 1
HIGH_DIM = N_STOCK > 1

if HIGH_DIM:
    COEF_ = 1e11
    S_OUTSTANDING = torch.tensor([1.15, 0.32, 0.23]) *1e10 / COEF_
    GAMMA = 1/(1/ (8.91*1e-13) + 1/ (4.45 * 1e-12) ) * COEF_
    BM_COV = torch.eye(3) #[[1, 0.5], [0.5, 1]]
else:
    COEF_ = 245714618646 # 1e11
    S_OUTSTANDING = torch.tensor([245714618646]) / COEF_ #torch.tensor([1.15, 0.32, 0.23]) *1e10 / COEF_
    GAMMA = 1.661728e-13 * COEF_ #1/(1/ (8.91*1e-13) + 1/ (4.45 * 1e-12) ) * COEF_
    BM_COV = torch.eye(1) #torch.eye(3) #[[1, 0.5], [0.5, 1]]
## END HERE ##

TIMESTAMPS = np.linspace(0, TR, T + 1)
#BM_COV = torch.tensor(BM_COV)
assert len(BM_COV.shape) == 2 and BM_COV.shape[0] == BM_COV.shape[1] and BM_COV.shape[0]
N_BM = BM_COV.shape[0]

## TODO: Adjust this function to get constant processes
## Compute constants processes using dW
def get_constants_high_dim(dW_std, W_s0d = None):
    time_len = dW_std.shape[1]
    n_sample = dW_std.shape[0]
    if W_s0d is not None:
        W_0 = W_s0d.reshape((n_sample, 1, N_BM)).cpu()
    else:
        W_0 = torch.zeros((n_sample, 1, N_BM))
    W_std = torch.cumsum(torch.cat((W_0, dW_std.cpu()), dim=1), dim=1)
    
    mu_stm = torch.ones((n_sample, time_len, N_STOCK)) * torch.tensor([[2.99, 3.71, 3.55]]) #torch.tensor([[2.99, 3.71, 3.55]]).repeat(T,1)
    #sigma_big = torch.tensor([[72.00, 71.49, 54.80],[71.49, 85.42, 65.86],[54.80, 65.86, 56.84]])
    sigma_big = torch.tensor([[72.00, 0, 0],[0, 85.42, 0],[0, 0, 56.84]])
    sigma_md = solve_sigma_md_theoretical(sigma_big) #torch.ones((T, N_STOCK, N_BM)) #???
    sigma_stmd = torch.ones((n_sample, time_len, N_STOCK, N_BM)) * sigma_md
    s_tm = torch.ones((time_len, N_STOCK))
    #xi_dd = torch.tensor([[ -2.07, 1.91, 0.64],[1.91, -1.77, -0.59],[0.64 ,-0.59 ,-0.20]]) *1e9 / COEF_
    xi_dd = torch.tensor([[2.07, 0, 0],[0, 1.77, 0],[0 , 0, -2.20]]) *1e9 / COEF_ * 100
    #lam_mm = torch.diag(torch.tensor([0.1269, 0.3354, 0.8595])) * 1e-8 * COEF_ * 0.01 #torch.ones((N_STOCK, N_STOCK))
    lam_mm = torch.diag(torch.tensor([0.1269, 0.1354, 0.1595])) * 1e-8 * COEF_ * 0.01 #torch.ones((N_STOCK, N_STOCK))
    alpha_md = sigma_md.clone() #torch.ones((N_STOCK, N_BM)) #???
    beta_m = torch.ones(N_STOCK) #???

    return W_std.to(device = DEVICE), mu_stm.to(device = DEVICE), sigma_stmd.to(device = DEVICE), s_tm.to(device = DEVICE), xi_dd.to(device = DEVICE), lam_mm.to(device = DEVICE), alpha_md.to(device = DEVICE), beta_m.to(device = DEVICE)

## Compute constants processes using dW
def get_constants_1_dim(dW_std, W_s0d = None):
    time_len = dW_std.shape[1]
    n_sample = dW_std.shape[0]
    if W_s0d is not None:
        W_0 = W_s0d.reshape((n_sample, 1, N_BM)).cpu()
    else:
        W_0 = torch.zeros((n_sample, 1, N_BM))
    W_std = torch.cumsum(torch.cat((W_0, dW_std.cpu()), dim=1), dim=1)
    
    mu_stm = torch.ones((n_sample, time_len, N_STOCK)) * 0.072068
    sigma_md = torch.tensor([[1.8788381]])
    sigma_stmd = torch.ones((n_sample, time_len, N_STOCK, N_BM)) * sigma_md
    s_tm = torch.ones((time_len, N_STOCK))
    if POWER == 2:
        xi_dd = torch.tensor([[2.19]]) * 1e10 / COEF_
        lam_mm = torch.diag(torch.tensor([1.08])) * 1e-10 * COEF_ * 1
    else: # POWER = 3/2
        xi_dd = torch.tensor([[2.33]]) * 1e10 / COEF_
        lam_mm = torch.diag(torch.tensor([5.22])) * 1e-6 * (COEF_ ** 0.5) * 1
    alpha_md = sigma_md.clone()
    beta_m = torch.ones(N_STOCK)

    return W_std.to(device = DEVICE), mu_stm.to(device = DEVICE), sigma_stmd.to(device = DEVICE), s_tm.to(device = DEVICE), xi_dd.to(device = DEVICE), lam_mm.to(device = DEVICE), alpha_md.to(device = DEVICE), beta_m.to(device = DEVICE)

def get_constants(dW_std, W_s0d = None):
    if HIGH_DIM:
        return get_constants_high_dim(dW_std, W_s0d)
    return get_constants_1_dim(dW_std, W_s0d)

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
torch.manual_seed(1)

#MULTI_NORMAL = MultivariateNormal(torch.zeros((N_SAMPLE, T, N_BM)), BM_COV * TR / T)
#dW_STD = MULTI_NORMAL.sample().to(device = DEVICE)
dW_STD = torch.randn([N_SAMPLE,N_BM,T])*np.sqrt(TR/T)
dW_STD = dW_STD[:,0,:].reshape((N_SAMPLE, T, N_BM))
W_STD, MU_STM, SIGMA_STMD, S_TM, XI_DD, LAM_MM, ALPHA_MD, BETA_M = get_constants(dW_STD)

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

        self.layer_lst.append(nn.Linear(INPUT_DIM, HIDDEN_DIM_LST[0], bias=True))
        self.bn.append(nn.BatchNorm1d(HIDDEN_DIM_LST[0],momentum=0.1))
        for i in range(1, len(HIDDEN_DIM_LST)):
            self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[i - 1], HIDDEN_DIM_LST[i], bias=True))
            self.bn.append(nn.BatchNorm1d(HIDDEN_DIM_LST[i],momentum=0.1))
        self.layer_lst.append(nn.Linear(HIDDEN_DIM_LST[-1], OUTPUT_DIM, bias=True))

    def forward(self, x):
        for i in range(len(self.layer_lst) - 1):
            x = self.layer_lst[i](x)
#            x = self.bn[i](x)
            x = F.relu(x)
        return self.layer_lst[-1](x)

class RL_Net(nn.Module):
    def __init__(self,INPUT_DIM,HIDDEN_DIM, OUTPUT_DIM = 1):
        super(RL_Net, self).__init__()
        self.input_dim = INPUT_DIM
        self.output_dim = OUTPUT_DIM
        self.hidden_dim = HIDDEN_DIM
        current_dim = self.input_dim
        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        for hdim in self.hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            self.bn.append(nn.BatchNorm1d(hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, self.output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.bn[i](x)
            x = F.relu(x)
        out = self.layers[-1](x)
        return out

## Model wrapper
class ModelFull(nn.Module):
    def __init__(self, predefined_model, is_discretized = False, pasting_ret = False, orig_len = None):
        super(ModelFull, self).__init__()
        self.model = predefined_model
        self.is_discretized = is_discretized
        self.pasting_ret = pasting_ret
        if orig_len is None:
            if is_discretized:
                self.orig_len = len(predefined_model)
            else:
                self.orig_len = 1
        else:
            self.orig_len = orig_len
    
    def forward(self, tup):
        t, x = tup
        if self.is_discretized:
            return self.model[t](x)
        else:
            return self.model(x)
    
    def get_pasting(self):
        if not self.pasting_ret:
            return ModelFull(self.model[1:], self.is_discretized, True, self.orig_len)
        return self
    
    def get_pasting_frac(self, size):
        return ModelFull(self.model[-size:], self.is_discretized, True, self.orig_len)

## Construct arbitrary neural network models with optimizer and scheduler
class ModelFactory:
    def __init__(self, algo, model_name, input_dim, hidden_lst, output_dim, lr, decay, scheduler_step, solver = "Adam", retrain = False, pasting_cutoff = 0, pasting_T = None):
        assert solver in ["Adam", "SGD", "RMSprop"]
        assert model_name in ["discretized_feedforward", "rnn"]
        assert algo in ["deep_hedging", "fbsde", "pasting"]
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
        self.pasting_cutoff = pasting_cutoff
        self.pasting_T = pasting_T

        if not retrain:
            self.model, self.prev_ts = self.load_latest()
            print(self.prev_ts)

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
        if self.algo == "pasting":
            if self.pasting_T is None:
                time_len = T - self.pasting_cutoff
            else:
                time_len = self.pasting_T
            time_len += 1
        else:
            time_len = T
        model_list = nn.ModuleList()
        for _ in range(time_len):
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
        torch.save(model_save, f"{drive_dir}/Models/{self.algo}_{self.model_name}__{curr_ts}.pt")
        self.model = self.model.to(device=DEVICE)
        return curr_ts
    
    def load_latest(self):
        ts_lst = [f.strip(".pt").split("__")[1] for f in os.listdir(f"{drive_dir}/Models/") if f.endswith(".pt") and f.startswith(self.algo+"_"+self.model_name)] #f.startswith(self.model_name)
        ts_lst = sorted(ts_lst, reverse=True)
        if len(ts_lst) == 0:
            return None, None
        ts = ts_lst[0]

        model = torch.load(f"{drive_dir}/Models/{self.algo}_{self.model_name}__{ts}.pt")
        model = model.to(device = DEVICE)
        return model, ts

## TODO: Implement it
## Return tensors of phi_dot and phi at each timestamp t
class DynamicsFactory():
    def __init__(self, ts_lst, dW_std, W_s0d = None, g_dir = None):
        assert ts_lst[0] == 0
        self.dW_std = dW_std
        self.n_sample = self.dW_std.shape[0]
        self.W_std, self.mu_stm, self.sigma_stmd, self.s_tm, self.xi_dd, self.lam_mm, self.alpha_md, self.beta_m = get_constants(dW_std, W_s0d)
        self.lam_mm_half = self.mat_frac_pow(self.lam_mm, 1/2)
        self.lam_mm_negHalf = self.mat_frac_pow(self.lam_mm, -1/2)
        self.alpha_mm_sq = self.alpha_md @ self.alpha_md.T
        self.const_mm = (GAMMA ** (1/2)) * self.mat_frac_pow(self.lam_mm_negHalf @ self.alpha_mm_sq @ self.lam_mm_negHalf, 1/2) #(GAMMA ** (1/2)) * self.lam_mm_negHalf @ self.mat_frac_pow(self.lam_mm_negHalf @ self.alpha_mm_sq @ self.lam_mm_negHalf, 1/2) @ self.lam_mm_half
        self.T = len(ts_lst) - 1
        self.ts_lst = ts_lst
        self.sigma_stmm_sq = torch.einsum("sijk, silk -> sijl", self.sigma_stmd, self.sigma_stmd)
        self.sigma_stmm_sq_inv = torch.zeros((self.n_sample, self.T, N_STOCK, N_STOCK)).to(device = DEVICE)
        self.sigma_mm_sq_inv = torch.inverse(self.sigma_stmm_sq[0,0,:,:])
        self.sigma_stmm_sq_inv += self.sigma_mm_sq_inv
#        for s in range(self.n_sample):
#            for t in range(self.T):
#                self.sigma_stmm_sq_inv[s,t,:,:] = torch.inverse(self.sigma_stmm_sq[s,t,:,:]) #self.sigma_stmm_sq_inv[t,:,:] #
        self.xi_std_w = torch.einsum("ijk, kl -> ijl", self.W_std[:,1:,:], self.xi_dd)
        self.phi_stm_bar = 1 / GAMMA * torch.einsum("sijk, sik -> sij", self.sigma_stmm_sq_inv, self.mu_stm) - torch.einsum("ijlk, ijk -> ijl", torch.einsum("sijk, sikl -> sijl", self.sigma_stmm_sq_inv, self.sigma_stmd), self.xi_std_w)
        if g_dir is not None:
            with open(g_dir, "r") as f:
                self.G_MAP = torch.tensor([float(x.strip()) for x in f.readlines()]).to(device=DEVICE)
    
    def get_constant_processes(self):
        return self.W_std, self.mu_stm, self.sigma_stmd, self.s_tm, self.xi_dd, self.lam_mm, self.alpha_md, self.beta_m
    
    def mat_frac_pow(self, mat, power):
        evals, evecs = torch.eig(mat, eigenvectors = True)
        mat_ret = torch.matmul(evecs, torch.matmul(torch.diag(evals[:,0] ** power), torch.inverse(evecs)))
        return mat_ret
    
    ## TODO: Implement it -- Daran Xu
    def fbsde_quad_highdim(self, model, dt = TR / T):
        # need (1 + T) models in total
        # model output should be
        Z_stmd = torch.zeros((self.n_sample, T, N_STOCK, N_BM)).to(device=DEVICE)
        phi_stm = torch.zeros((self.n_sample, T + 1, N_STOCK)).to(device=DEVICE)
        phi_stm[:, 0, :] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        phi_dot_stm = torch.zeros((self.n_sample, T + 1, N_STOCK)).to(device=DEVICE)  # note here phi_dot has T+1 timesteps
        curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        phi_dot_stm[:, 0, :] = model((-1, curr_t)) #curr_t as dummy input
        INV_LAMBDA_ON_100=torch.inverse(100*self.lam_mm)
        for t in range(T):            
            x = torch.cat((self.W_std[:, t, :], (t+1) / T * TR * curr_t), dim=1).to(device=DEVICE)
            Z_stmd[:, t, :, :] = model((t, x)).reshape(self.n_sample, N_STOCK, N_BM)
            phi_dot_stm[:, t + 1, :] = phi_dot_stm[:, t, :] + \
                + 100*TR / T * GAMMA * torch.einsum("bij,bj -> bi",
                                               torch.einsum("bij,bkj -> bik",
                                                    torch.einsum("ji ,bik->bjk",
                                                                INV_LAMBDA_ON_100, self.sigma_stmd[:,t, :, :]),
                                                            self.sigma_stmd[:,t, :, :]) ,
                                                (phi_stm[:, t, :] - self.phi_stm_bar[:, t, :]) 
                                               ) + \
                +torch.einsum('bik, bk -> bi', Z_stmd[:, t, :, :], self.dW_std[:, t, :])       
            phi_stm[:, t + 1, :] = phi_stm[:, t, :] + phi_dot_stm[:, t+1, :] * (self.ts_lst[t + 1] - self.ts_lst[t])
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
    
    def fbsde_quad0(self, model, time_len = T, phi_0 = None, start_t = 0, dt = TR / T):
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        phi_dot_stm[:,0,:] = model((-1, curr_t)) #curr_t as dummy input
        d_phi_dot_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        for t in range(time_len):
            x = phi_stm[:,t,:] - self.phi_stm_bar[:,start_t + t,:]
            y = model((t, x))
            d_phi_dot_stm[:,t,:] = GAMMA * torch.einsum("ijk -> ij", self.sigma_stmd[:,t+start_t,:,:]) / torch.sum(self.lam_mm) * x * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t]) + y * self.dW_std[:,t+start_t,:]
            phi_dot_stm[:,t+1,:] = phi_dot_stm[:,t,:] + d_phi_dot_stm[:,t,:]
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t+1,:] * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t])
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- TBD
    def fbsde_power0(self, model, time_len = T, phi_0 = None, start_t = 0, dt = TR / T):
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        y = model((-1, curr_t))
        phi_dot_stm[:,0,:] = y #curr_t as dummy input
        Y_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        Y_stm[:,0,:] = torch.sum(self.lam_mm) * torch.sign(y) * torch.abs(y) ** 0.5
        for t in range(time_len):
            x = phi_stm[:,t,:] - self.phi_stm_bar[:,start_t + t,:]
            y = model((t, x))
            phi_dot_stm[:,t+1,:] = torch.sign(Y_stm[:,t,:]) * Y_stm.clone()[:,t,:] ** 2 / torch.sum(self.lam_mm) ** 2
            dt = self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t]
            Y_stm[:,t+1,:] = Y_stm[:,t,:] + GAMMA * torch.einsum("ijk -> ij", self.sigma_stmd[:,t+start_t,:,:]) * x * dt + y * self.dW_std[:,t+start_t,:]
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * dt
        return phi_dot_stm, phi_stm, Y_stm
    
    def fbsde_quad(self, model, time_len = T, phi_0 = None, start_t = 0, dt = TR / T):
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len, N_STOCK)).to(device = DEVICE)
        curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        delta_phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        Y_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        Y_stm[:,0,:] = model((-1, curr_t))
        delta_phi_stm[:,0,:] = phi_stm[:,0,:] - self.phi_stm_bar[:,start_t,:]
        for t in range(time_len):
            x = phi_stm[:,t,:] - self.phi_stm_bar[:,start_t + t,:]
            y = model((t, x))
            val = (Y_stm[:,t,:] / torch.sum(self.lam_mm)) * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t]) + torch.sum(self.xi_dd) / torch.einsum("ijk -> ij", self.sigma_stmd[:,t+start_t,:,:]) * self.dW_std[:,t+start_t,:]
            delta_phi_stm[:,t+1,:] = delta_phi_stm[:,t,:] + val
            dY = GAMMA * torch.einsum("ijk -> ij", self.sigma_stmm_sq[:,t+start_t,:,:]) * delta_phi_stm[:,t,:] * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t]) + y * self.dW_std[:,t+start_t,:]
            Y_stm[:,t+1,:] = Y_stm[:,t,:] + dY
            phi_dot_stm[:,t,:] = Y_stm[:,t,:] / torch.sum(self.lam_mm)
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t])
        return phi_dot_stm, phi_stm, Y_stm
    
    def fbsde_power(self, model, time_len = T, phi_0 = None, start_t = 0, dt = TR / T):
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len, N_STOCK)).to(device = DEVICE)
        curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        delta_phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        Y_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        Y_stm[:,0,:] = model((-1, curr_t))
        delta_phi_stm[:,0,:] = phi_stm[:,0,:] - self.phi_stm_bar[:,start_t,:]
        for t in range(time_len):
            x = phi_stm[:,t,:] - self.phi_stm_bar[:,start_t + t,:]
            y = model((t, x))
            val = (Y_stm[:,t,:] / torch.sum(self.lam_mm)) ** 2 * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t]) + torch.sum(self.xi_dd) / torch.einsum("ijk -> ij", self.sigma_stmd[:,t+start_t,:,:]) * self.dW_std[:,t+start_t,:]
            delta_phi_stm[:,t+1,:] = delta_phi_stm[:,t,:] + val
            dY = GAMMA * torch.einsum("ijk -> ij", self.sigma_stmm_sq[:,t+start_t,:,:]) * delta_phi_stm[:,t,:] * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t]) + y * self.dW_std[:,t+start_t,:]
            Y_stm[:,t+1,:] = Y_stm[:,t,:] + dY
            phi_dot_stm[:,t,:] = (Y_stm[:,t,:] / torch.sum(self.lam_mm)) ** 2
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t])
        return phi_dot_stm, phi_stm, Y_stm
    
    ## TODO: Implement it -- Zhanhao Zhang
    def leading_order_quad(self, model = None, time_len = None, phi_0 = None, dt = TR / T):
        if time_len is None:
            time_len = self.T
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len, N_STOCK)).to(device = DEVICE)
        for t in range(time_len):
            phi_dot_stm[:,t,:] = -(phi_stm[:,t,:] - self.phi_stm_bar[:,t,:]) @ self.lam_mm_negHalf @ self.const_mm @ self.lam_mm_half
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + 1] - self.ts_lst[t])
        return phi_dot_stm, phi_stm
    
    def g_vec(self, x):
        q = 3 / 2
        x_ind = torch.round((torch.abs(x) + 0) / 50 * 500000).long()
        x_inbound = (torch.abs(x) <= 50) + 0
        x_outbound = -torch.sign(x) * q * (q - 1) ** (-(q - 1) / q) * torch.abs(x) ** (2 * (q - 1) / q)
        return torch.sign(x) * self.G_MAP[x_ind * x_inbound] + x_outbound * (1 - x_inbound)
    
    ## TODO: Implement it -- Zhanhao Zhang
    def leading_order_power(self, power, model = None, time_len = None, phi_0 = None, dt = TR / T):
        # Currently only support 1-dim
        if time_len is None:
            time_len = self.T
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len, N_STOCK)).to(device = DEVICE)
        for t in range(time_len):
            phi_sm_minus_bar = phi_stm[:,t,:] - self.phi_stm_bar[:,t,:]
            outer = -torch.sign(phi_sm_minus_bar) * (power * GAMMA * torch.sum(self.xi_dd) ** 4 / 8 / torch.sum(self.lam_mm) / torch.sum(self.alpha_md) ** 2) ** (1 / (power + 2))
            inner = 2 ** ((power - 1) / (power + 2)) * ((power * GAMMA * torch.sum(self.alpha_md) ** 2 / torch.sum(self.lam_mm)) ** (1 / (power + 2))) * ((torch.sum(self.alpha_md) / torch.sum(self.xi_dd)) ** (2 * power / (power + 2))) * phi_sm_minus_bar
            phi_dot_stm[:,t,:] = outer * torch.abs(self.g_vec(inner) / power)
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + 1] - self.ts_lst[t])
        return phi_dot_stm, phi_stm
    
    def leading_order_power_highdim(self, power, model = None, time_len = None, phi_0 = None, dt = TR / T):
        # Tentative
        if time_len is None:
            time_len = self.T
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len, N_STOCK)).to(device = DEVICE)
        gamma_alpha_lam_mm = self.mat_frac_pow(self.lam_mm_negHalf @ self.const_mm @ self.lam_mm_half, 2)
        alpha_xi_md = self.alpha_md @ torch.inverse(self.xi_dd)
        for t in range(time_len):
            phi_sm_minus_bar = phi_stm[:,t,:] - self.phi_stm_bar[:,t,:]
            outer = -torch.sign(phi_sm_minus_bar) @ self.mat_frac_pow(power * gamma_alpha_lam_mm @ self.mat_frac_pow(torch.abs(alpha_xi_md), -4) * torch.sign(alpha_xi_md), (1 / (power + 2)))
            inner = phi_sm_minus_bar @ (2 ** ((power - 1) / (power + 2)) * (self.mat_frac_pow(power * gamma_alpha_lam_mm, 1 / (power + 2))) @ self.mat_frac_pow(alpha_xi_md, 2 * power / (power + 2)))
            phi_dot_stm[:,t,:] = outer * torch.abs(self.g_vec(inner) / power)
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + 1] - self.ts_lst[t])
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- Zhanhao Zhang
    def ground_truth(self, model = None, phi_0 = None, dt = TR / T):
        phi_stm = torch.zeros((self.n_sample, self.T + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, self.T, N_STOCK)).to(device = DEVICE)
        for t in range(self.T):
            tanh_inner = self.const_mm / T * (self.T - t - 1) * TR
            evals, evecs = torch.eig(tanh_inner, eigenvectors = True)
            norm_fac = torch.sum(evals[:,0] ** 2) ** 0.5 + 1e-9
            tanh_exp = torch.matmul(evecs, torch.matmul(torch.diag(torch.exp(evals[:,0] / norm_fac)), torch.inverse(evecs)))
            tanh_exp_neg = torch.matmul(evecs, torch.matmul(torch.diag(torch.exp(evals[:,0] / norm_fac * (1 - 2 * norm_fac))), torch.inverse(evecs)))
            tanh_tmp = torch.inverse(tanh_exp_neg + tanh_exp) @ (tanh_exp - tanh_exp_neg) #1 - 2 * torch.inverse(tanh_exp + 1)
            phi_dot_stm[:,t,:] = -(phi_stm[:,t,:] - self.phi_stm_bar[:,t,:]) @ self.lam_mm_negHalf @ self.const_mm @ self.lam_mm_half @ tanh_tmp #torch.tanh(self.const_mm * (T - t) / T * TR).T
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + 1] - self.ts_lst[t])
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- Zhanhao Zhang
    def deep_hedging_own(self, model, time_len = T, phi_0 = None, start_t = 0, dt = TR / T, orig=False):
        phi_stm = torch.zeros((self.n_sample, time_len + 1, N_STOCK)).to(device = DEVICE)
        if phi_0 is None:
            phi_stm[:,0,:] = self.phi_stm_bar[:,0,:] #S_OUTSTANDING / 2
        else:
            phi_stm[:,0,:] = phi_0
        phi_dot_stm = torch.zeros((self.n_sample, time_len, N_STOCK)).to(device = DEVICE)
        curr_t = torch.ones((self.n_sample, 1)).to(device = DEVICE)
        xi = torch.sum(self.xi_dd)
        for t in range(time_len):
            #x = torch.cat((phi_stm[:,t,:], self.W_std[:,t + start_t,:], curr_t), dim = 1)#.to(device = DEVICE)
            t_coef = (model.orig_len - time_len + t) #* 0.25
#            if False: #POWER == 2:
#                x = torch.cat((phi_stm[:,t,:], xi * self.W_std[:,t + start_t,:], curr_t * t_coef, 0.072068 * torch.ones((self.n_sample, 1)).to(device = DEVICE), 1.8788381 * torch.ones((self.n_sample, 1)).to(device = DEVICE)), dim = 1)
#            else:
#                x = torch.cat((phi_stm[:,t,:], xi * self.W_std[:,t + start_t,:], curr_t * t_coef), dim = 1)
            x = phi_stm[:,t,:] - self.phi_stm_bar[:,start_t + t,:]
            t_tensor= torch.ones(self.n_sample) * t / T * TR
            x2=torch.cat((phi_stm[:,t,0].reshape(-1,1), xi * self.W_std[:,t,0].reshape(-1,1), t_tensor.reshape(-1,1)),dim=1).to(device=DEVICE)
            if orig:
                x = x2
            phi_dot_stm[:,t,:] = model((t, x))
#            if orig:
#                phi_dot_stm[:,t,:] /= 1000
#            phi_dot_stm[:,t,-1] = -torch.sum(phi_dot_stm[:,t,:-1])
            phi_stm[:,t+1,:] = phi_stm[:,t,:] + phi_dot_stm[:,t,:] * (self.ts_lst[t + start_t + 1] - self.ts_lst[t + start_t])
        return phi_dot_stm, phi_stm
    
    def deep_hedging(self, model, time_len = T, phi_0 = None, start_t = 0, dt = TR / T, orig=False):
        N = self.n_sample
        PHI_0_on_s = torch.ones(N)/2
        W=torch.cumsum(self.dW_std[:,:,0], dim=1) #ttttt
        W=torch.cat((torch.zeros((N,1)),W),dim=1)  #(N,T+1)
        XI = torch.sum(self.xi_dd)
        XI_W_on_s = XI* W
        ### UTILITY
        PHI_on_s = torch.zeros((N, T + 1)).to(device=DEVICE)
        PHI_on_s[:,0] = PHI_on_s[:,0]+PHI_0_on_s.reshape((-1,))
        PHI_dot_on_s = torch.zeros((N, T )).to(device=DEVICE)

        for t in range(T):
            ### UTILITY
            t_tensor=t/T*TR*torch.ones(N).reshape(-1,1).to(device=DEVICE)
            x_Utility=torch.cat((PHI_on_s[:,t].reshape(-1,1),XI_W_on_s[:,t].reshape(-1,1),t_tensor),dim=1)
            PHI_dot_on_s[:,t] = model((t, x_Utility)).reshape(-1,)
            PHI_on_s[:,(t+1)] = PHI_on_s[:,t].reshape(-1)+PHI_dot_on_s[:,(t)].reshape(-1)
        return PHI_dot_on_s.reshape((N, T, 1)), PHI_on_s.reshape((N, T+1, 1))
    
    ## TODO: Implement it -- Zhanhao Zhang
    def random_deep_hedging(self, model, time_len = T, phi_0 = None, dt = TR / T):
#        if phi_0 is None:
#            phi_0 = torch.tensor([0.0617]) #torch.tensor([0.1603, -0.7572,  1.5443]) #torch.rand(N_STOCK) * 2
        phi_dot_stm, phi_stm = self.deep_hedging(model, time_len = time_len, phi_0 = phi_0, dt = dt)
        return phi_dot_stm, phi_stm
    
    ## TODO: Implement it -- Zhanhao Zhang
    def pasting(self, model, M, dt = TR / T, pasting_T = None, pasting_algo = "deep_hedging"):
#        phi_stm = torch.zeros((self.n_sample, T + 1, N_STOCK)).to(device = DEVICE)
##         phi_stm[:,0,:] = S_OUTSTANDING / 2
#        phi_dot_stm = torch.zeros((self.n_sample, T, N_STOCK)).to(device = DEVICE)
        if POWER == 2:
            phi_dot_stm_leading_order, phi_stm_leading_order = self.leading_order_quad(time_len = M)
        else:
            phi_dot_stm_leading_order, phi_stm_leading_order = self.leading_order_power(POWER, time_len = M)
#        print(phi_stm_leading_order[0,-1,:])
        if pasting_T is None:
            time_len = T - M
        else:
            time_len = pasting_T
        if pasting_algo == "deep_hedging":
            phi_dot_stm_pasting, phi_stm_pasting = self.deep_hedging_own(model, time_len = time_len, phi_0 = phi_stm_leading_order[:,-1,:].data, start_t = M, dt = dt)
        else:
            if POWER == 2:
                phi_dot_stm_pasting, phi_stm_pasting, _ = self.fbsde_quad(model, time_len = time_len, phi_0 = phi_stm_leading_order[:,-1,:].data, start_t = M, dt = dt)
            else:
                phi_dot_stm_pasting, phi_stm_pasting = self.fbsde_power(model, time_len = time_len, phi_0 = phi_stm_leading_order[:,-1,:].data, start_t = M, dt = dt)
#        phi_dot_stm[:,:M,:] += phi_dot_stm_leading_order
#        phi_stm[:,:(M+1),:] += phi_stm_leading_order
#        phi_dot_stm[:,M:,:] += phi_dot_stm_deep_hedging
#        phi_stm[:,(M+1):,:] += phi_stm_deep_hedging[:,1:,:]
        phi_dot_stm = torch.cat((phi_dot_stm_leading_order, phi_dot_stm_pasting), dim = 1)
        phi_stm = torch.cat((phi_stm_leading_order, phi_stm_pasting[:,1:,:]), dim = 1)
        return phi_dot_stm, phi_stm

## TODO: Implement it
## Return the loss as a tensor
class LossFactory():
    def __init__(self, ts_lst, dW_std, W_s0d = None):
        assert ts_lst[0] == 0
        self.dW_std = dW_std
        self.ts_lst = ts_lst
        self.dt_lst = self.ts_lst[1:] - self.ts_lst[:-1]
        self.TR = self.ts_lst[-1] - self.ts_lst[0]
        self.dt_lst = torch.tensor(self.dt_lst).reshape((len(self.dt_lst), 1)).float().to(device=DEVICE)
        self.W_std, self.mu_stm, self.sigma_stmd, self.s_tm, self.xi_dd, self.lam_mm, self.alpha_md, self.beta_m = get_constants(dW_std, W_s0d)
        self.n_sample = self.dW_std.shape[0]
        self.mse_loss_func=torch.nn.MSELoss()

    ## TODO: Implement it -- Zhanhao Zhang
    def utility_loss(self, phi_dot_stm, phi_stm, power, is_arr = False):
#        if power == 2:
#            loss_mat = torch.einsum("ijk, ijk -> ij", phi_stm[:,1:,:], self.mu_stm) - GAMMA / 2 * torch.einsum("ijk -> ij", (torch.einsum("ijk, ijkl -> ijl", phi_stm[:,1:,:], self.sigma_stmd) + torch.einsum("ijk, kl -> ijl", self.W_std[:,1:,:], self.xi_dd)) ** 2) - 1 / 2 * torch.einsum("ijk, lk, ijl -> ij", phi_dot_stm, self.lam_mm, phi_dot_stm)
#        else:
#            ## Currently only support 1-dim.
#            loss_mat = torch.einsum("ijk, ijk -> ij", phi_stm[:,1:,:], self.mu_stm) - GAMMA / 2 * torch.einsum("ijk -> ij", (torch.einsum("ijk, ijkl -> ijl", phi_stm[:,1:,:], self.sigma_stmd) + torch.einsum("ijk, kl -> ijl", self.W_std[:,1:,:], self.xi_dd)) ** 2) - 1 / power * torch.einsum("ijk, lk -> ij", torch.abs(phi_dot_stm) ** power, self.lam_mm)
        loss_mat = torch.einsum("ijk, ijk -> ij", phi_stm[:,1:,:], self.mu_stm) - GAMMA / 2 * torch.einsum("ijk -> ij", (torch.einsum("ijk, ijkl -> ijl", phi_stm[:,1:,:], self.sigma_stmd) + torch.einsum("ijk, kl -> ijl", self.W_std[:,1:,:], self.xi_dd)) ** 2) - 1 / 2 * torch.einsum("ijk, lk, ijl -> ij", torch.abs(phi_dot_stm) ** (POWER / 2), self.lam_mm, torch.abs(phi_dot_stm) ** (POWER / 2))
        if not is_arr:
            loss_compact = -torch.sum(torch.einsum("ij, jk -> ij", loss_mat / self.n_sample, self.dt_lst)) / self.TR
        else:
            loss_compact = -torch.sum(torch.einsum("ij, jk -> ij", loss_mat / self.n_sample, self.dt_lst), axis = 0) / self.TR
        return loss_compact
    
    def utility_loss_report(self, phi_dot_stm, phi_stm, power):
#        loss_mat = torch.einsum("ijk, ijk -> ij", phi_stm[:,1:,:], self.mu_stm) - GAMMA / 2 * torch.einsum("ijk -> ij", (torch.einsum("ijk, ijkl -> ijl", phi_stm[:,1:,:], self.sigma_stmd) + torch.einsum("ijk, kl -> ijl", self.W_std[:,1:,:], self.xi_dd)) ** 2) - 1 / 2 * torch.einsum("ijk, lk, ijl -> ij", torch.abs(phi_dot_stm) ** (POWER / 2), self.lam_mm, torch.abs(phi_dot_stm) ** (POWER / 2))
#        utility_arr = torch.einsum("ij, jk -> ij", loss_mat, self.dt_lst) / self.TR
#        utility_arr = utility_arr.data.numpy()
#        mu = np.sum(utility_arr) / self.n_sample
#        std = np.sqrt(np.sum(utility_arr ** 2) / self.n_sample)
#        term_err = np.mean(phi_dot_stm[:,-1,:].data.numpy() ** 2, axis = 0)
        C_1 = GAMMA / 2
        C_2 = torch.sum(self.lam_mm) / 2
        N = self.n_sample
        MU_BAR = self.mu_stm[0,0,0]
        ALPHA = 1.8788381
        PHI_on_s = phi_stm[:,:,0]
        PHI_dot_on_s = phi_dot_stm[:,:,0]
        W=torch.cumsum(self.dW_std[:,:,0], dim=1) #ttttt
        W=torch.cat((torch.zeros((N,1)),W),dim=1)  #(N,T+1)
        XI = torch.sum(self.xi_dd)
        XI_W_on_s = XI* W
        loss = -torch.sum(MU_BAR*PHI_on_s[:,:-1],1) + C_1*torch.sum((ALPHA*PHI_on_s[:,:-1]+XI_W_on_s[:,:-1])**2,1) + C_2*torch.sum((torch.abs(PHI_dot_on_s))**(2),1)
        loss = loss * TR / T
        loss = -loss.data.numpy()
        mu = np.mean(loss) / TR
        std = np.std(loss) / TR
        term_err = np.mean(phi_dot_stm[:,-1,:].data.numpy() ** 2, axis = 0)
        return mu, std, term_err
    
    def utility_loss_report2(self, phi_dot_stm, phi_stm, power):
        C_1 = GAMMA / 2
        C_2 = torch.sum(self.lam_mm) / 2
        N = self.n_sample
        MU_BAR = self.mu_stm[0,0,0]
        ALPHA = 1.8788381
        PHI_on_s = phi_stm[:,:,0]
        PHI_dot_on_s = phi_dot_stm[:,:,0]
        W=torch.cumsum(self.dW_std[:,:,0], dim=1) #ttttt
        W=torch.cat((torch.zeros((N,1)),W),dim=1)  #(N,T+1)
        XI = torch.sum(self.xi_dd)
        XI_W_on_s = XI* W
        loss = -torch.sum(MU_BAR*PHI_on_s[:,:-1],1) + C_1*torch.sum((ALPHA*PHI_on_s[:,:-1]+XI_W_on_s[:,:-1])**2,1) + C_2*torch.sum((torch.abs(PHI_dot_on_s))**(2),1)
        loss = loss / T
        loss = -loss.data.numpy()
        m1 = np.sum(loss) / self.n_sample
        m2 = np.sum(loss ** 2) / self.n_sample
        term_sum = np.sum(phi_dot_stm[:,-1,:].data.numpy() ** 2) / self.n_sample
        return m1, m2, term_sum
    
    ## TODO: Implement it -- Zhanhao Zhang
    def fbsde_loss(self, phi_dot_stm, phi_stm):
        Value=phi_dot_stm[:,-1,:]
        Target=torch.zeros_like(Value,device=DEVICE)
        return self.mse_loss_func(Value,Target) #torch.mean(torch.abs(Value - Target)) #


## Write training logs to file
def write_logs(ts_lst, train_args):
    with open(f"{drive_dir}/Logs.tsv", "a") as f:
        for i in range(1, len(ts_lst)):
            line = f"{ts_lst[i - 1]}\t{ts_lst[i]}\t{json.dumps(train_args)}\n"
            f.write(line)

## Visualize loss function through training
def visualize_loss(loss_arr, ts, loss_truth):
    plt.plot(loss_arr)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(y = loss_truth, color = "red")
    plt.title(f"Final Loss = {loss_arr[-1]:.2e}, True Loss = {loss_truth:.2e}")
    plt.savefig(f"{drive_dir}/Plots/loss_{ts}.png")
    plt.close()

## Visualize dynamics of an arbitrary component
def Visualize_dyn(timestamps, arr, ts, name):
    assert name in ["phi", "phi_dot", "sigma", "mu", "s"]
    if POWER == 2:
        suffix = "quad"
    else:
        suffix = "power"
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
    plt.savefig(f"{drive_dir}/Plots/{suffix}_{name}_{ts}.png")
    plt.close()

## Visualize the comparision of dynamics from different algorithms
def Visualize_dyn_comp(timestamps, arr_lst, ts, name, algo_lst, comment = None):
    assert name in ["phi", "phi_dot", "sigma", "mu", "s"]
    if POWER == 2:
        suffix = "quad"
    else:
        suffix = "power"
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
    if comment is not None:
        title2 = title + "\n" + str(comment)
    else:
        title2 = title
    if name == "phi_dot":
        size = arr_lst[0].cpu().detach().numpy().shape
        if len(size) == 1:
            size = 1
        else:
            size = size[1]
        for i in range(size):
            for arr, algo in zip(arr_lst, algo_lst):
                arr = arr.cpu().detach().numpy()
                if algo == "pasting":
                    algo = "ST-Hedging"
                if len(arr.shape) == 2 and arr.shape[1] == 1:
                    arr = arr.reshape((-1,))
                if len(arr.shape) == 1:
                    plt.plot(timestamps, arr,label = f"{algo}")
                else:
                    plt.plot(timestamps, arr[:, i], label = f"{algo} - Stock {i + 1}")
            plt.xlabel("T")
            plt.ylabel(title)
            plt.title(title2)
            plt.legend()
            plt.savefig(f"{drive_dir}/Plots/{suffix}_comp_{name}_stock{i+1}_{ts}.png")
            plt.close()
    else:
        for arr, algo in zip(arr_lst, algo_lst):
            arr = arr.cpu().detach().numpy()
            if algo == "pasting":
                algo = "ST-Hedging"
            if len(arr.shape) == 2 and arr.shape[1] == 1:
                arr = arr.reshape((-1,))
            if len(arr.shape) == 1:
                plt.plot(timestamps, arr,label = f"{algo}")
            else:
                for i in range(arr.shape[1]):
                    plt.plot(timestamps, arr[:, i], label = f"{algo} - Stock {i + 1}")
        plt.xlabel("T")
        plt.ylabel(title)
        plt.title(title2)
        plt.legend()
        plt.savefig(f"{drive_dir}/Plots/{suffix}_comp_{name}_{ts}.png")
        plt.close()

def Visualize_dyn_comp2(timestamps, arr_lst_phidot, arr_lst_phi, ts, name, algo_lst, comment = None):
    for i in range(len(algo_lst)):
        if algo_lst[i] == "pasting":
            algo_lst[i] = "ST-Hedging"
        else:
            algo_lst[i] = algo_lst[i].replace("_", " ").title()
    fig = plt.figure(figsize=(10,4))

    ax1=plt.subplot(1,2,1)
    L = len(arr_lst_phidot)
    for i in range(L):
        ax1.plot(timestamps, arr_lst_phidot[i].cpu().detach().numpy(), label = algo_lst[i])
    ax1.hlines(0,xmin=0,xmax=timestamps[-1],linestyles='dotted')

    ax1.title.set_text("{}".format("$\\dot{\\varphi_t}$"))
    ax1.grid()

    ax2=plt.subplot(1,2,2)
    for i in range(L):
        ax2.plot(timestamps,arr_lst_phi[i].cpu().detach().numpy(), label = algo_lst[i])
    ax2.title.set_text("{}".format("${\\varphi_t}$"))
    ax2.grid()
    box2=ax2.get_position()

    ax2.legend(loc="lower left", bbox_to_anchor=(box2.width*3,box2.height))

    plt.savefig(f"{drive_dir}/Plots/{name}.pdf", bbox_inches='tight')
    plt.close()

## The training pipeline
def training_pipeline(algo = "deep_hedging", cost = "quadratic", model_name = "discretized_feedforward", solver = "Adam", hidden_lst = [50], lr = 1e-2, epoch = 1000, decay = 0.1, scheduler_step = 10000, retrain = False, pasting_cutoff = 0, n_sample = N_SAMPLE, pasting_T = None, pasting_algo = "deep_hedging", train_freq = 1, train_cut = 10, **kargs):
    assert algo in ["deep_hedging", "fbsde", "pasting"]
    assert pasting_algo in ["deep_hedging", "fbsde"]
    assert cost in ["quadratic", "power"]
    assert model_name in ["discretized_feedforward", "rnn"]
    assert solver in ["Adam", "SGD", "RMSprop"]
    assert int(train_freq) >= 1
    train_freq = int(train_freq)
    
    if cost == "quadratic":
        power = 2
    else:
        power = 3 / 2
    
    ## TODO: Change the input dimension accordingly
    phi_0_leading = None
    if algo == "deep_hedging":
        output_dim = N_STOCK
        input_dim = N_STOCK #N_BM + N_STOCK + 1
    elif algo == "fbsde":
        output_dim = N_BM * N_STOCK
        input_dim = N_STOCK #+ 1
    else:
        output_dim = N_STOCK
        input_dim = N_STOCK #N_BM + N_STOCK + 1
    if algo == "pasting":
#        multi_normal = MultivariateNormal(torch.zeros((n_sample, T - pasting_cutoff, N_BM)), BM_COV * TR / T)
        dt = (T - pasting_cutoff) / T * TR / pasting_T
        multi_normal = MultivariateNormal(torch.zeros((n_sample, pasting_T + 1, N_BM)), BM_COV * dt)
#        if n_sample != N_SAMPLE:
#        torch.manual_seed(0)
#        MULTI_NORMAL_CURR = MultivariateNormal(torch.zeros((n_sample, train_cut, N_BM)), BM_COV * TR / T)
#        MULTI_NORMAL_CURR = MultivariateNormal(torch.zeros((n_sample, T, N_BM)), BM_COV * TR / T)
#        dW_STD_curr = MULTI_NORMAL_CURR.sample().to(device = DEVICE)
##        dW_STD_curr[:,:pasting_cutoff,:] = dW_STD_curr[0,:pasting_cutoff,:]
#        W_STD_curr, _, _, _, _, _, _, _ = get_constants(dW_STD_curr)
##        else:
##            dW_STD_curr, W_STD_curr = dW_STD, W_STD
#
#        cut = min(train_cut, pasting_cutoff)
#        W_s0d = W_STD_curr[:, cut + 1, :].reshape((n_sample, 1, N_BM)).cpu()
#
        if pasting_T is None:
            time_lst = TIMESTAMPS[:(T - pasting_cutoff + 1 + 1)]
        else:
            time_lst = np.linspace(0, (T - pasting_cutoff) / T * TR + dt, pasting_T + 1 + 1)
#        dynamic_factory = DynamicsFactory(TIMESTAMPS, dW_STD_curr, None, g_dir = f"{drive_dir}/eva.txt")
#        if cost == "quadratic":
#            phi_dot_stm_leading_order, phi_stm_leading_order = dynamic_factory.leading_order_quad()
#        else:
#            phi_dot_stm_leading_order, phi_stm_leading_order = dynamic_factory.leading_order_power(POWER)
#        phi_0_leading = phi_stm_leading_order[:, cut + 1,:]
    else:
        multi_normal = MULTI_NORMAL
        time_lst = TIMESTAMPS
    model_factory = ModelFactory(algo, model_name, input_dim, hidden_lst, output_dim, lr, decay, scheduler_step, solver, retrain, pasting_cutoff, pasting_T)

    model, optimizer, scheduler, prev_ts = model_factory.prepare_model()
    loss_arr = []
    loss_tmp = torch.tensor(0.).to(device=DEVICE)
    
    for itr in tqdm(range(epoch)):
        optimizer.zero_grad()
        ## TODO: Implement it
        dW_std = multi_normal.sample().to(device = DEVICE)
#        if algo == "pasting":
#            dW_STD_curr = MULTI_NORMAL_CURR.sample().to(device = DEVICE)
#            W_STD_curr, _, _, _, _, _, _, _ = get_constants(dW_STD_curr)
#            cut = min(train_cut, pasting_cutoff)
#            W_s0d = W_STD_curr[:, -1, :].reshape((n_sample, 1, N_BM)).cpu()
#
#            dynamic_factory2 = DynamicsFactory(TIMESTAMPS[:(train_cut + 1)], dW_STD_curr, None, g_dir = f"{drive_dir}/eva.txt")
#            if cost == "quadratic":
#                phi_dot_stm_leading_order, phi_stm_leading_order = dynamic_factory2.leading_order_quad()
#            else:
#                phi_dot_stm_leading_order, phi_stm_leading_order = dynamic_factory2.leading_order_power(POWER)
#            phi_0_leading = phi_stm_leading_order[:, cut,:]
#        else:
        W_s0d = None
        phi_0_leading = None
        dynamic_factory = DynamicsFactory(time_lst, dW_std, W_s0d, g_dir = f"{drive_dir}/eva.txt")
        loss_factory = LossFactory(time_lst, dW_std, W_s0d)

        if algo == "deep_hedging":
            phi_dot_stm, phi_stm = dynamic_factory.deep_hedging(model)
            loss = loss_factory.utility_loss(phi_dot_stm, phi_stm, power)
        elif algo == "fbsde":
            if cost == "quadratic":
                phi_dot_stm, phi_stm, Y_stm = dynamic_factory.fbsde_quad(model)
                loss = loss_factory.fbsde_loss(Y_stm, phi_stm)
            else:
                phi_dot_stm, phi_stm, Y_stm = dynamic_factory.fbsde_power(model)
                loss = loss_factory.fbsde_loss(Y_stm, phi_stm)
        else: #if algo == "pasting":
            phi_dot_stm, phi_stm = dynamic_factory.random_deep_hedging(model, pasting_T + 1, phi_0 = phi_0_leading, dt = dt)
            loss = loss_factory.utility_loss(phi_dot_stm, phi_stm, power)
        ## End here ##
        loss_tmp += loss / train_freq
        if itr % train_freq == train_freq - 1:
            loss_arr.append(float(loss_tmp.data))
            loss_tmp.backward()

            if torch.isnan(loss_tmp.data):
                break
            optimizer.step()
            scheduler.step()
            
            loss_tmp = torch.tensor(0.).to(device=DEVICE)
    
    if epoch > 0:
        ## Update and save model
        model_factory.update_model(model)
        curr_ts = model_factory.save_to_file()
        if algo == "pasting":
            model = model.get_pasting()

        ## Compute Ground Truth
        if cost == "quadratic":
            phi_dot_stm_ground_truth, phi_stm_ground_truth, loss_truth = evaluation(dW_std, curr_ts, None, algo = "ground_truth", cost = cost, visualize_obs = 0, phi_0 = phi_stm[:,0,:], W_s0d = W_s0d, time_lst = time_lst)
        phi_dot_stm_leading_order, phi_stm_leading_order, loss_leading_order = evaluation(dW_std, curr_ts, None, algo = "leading_order", cost = cost, visualize_obs = 0, phi_0 = phi_stm[:,0,:], W_s0d = W_s0d, time_lst = time_lst)

        ## Visualize loss and results
        if algo == "fbsde":
            loss_truth, loss_leading_order = 0, 0
        if cost == "quadratic":
            visualize_loss(loss_arr, curr_ts, loss_truth)
        else:
            visualize_loss(loss_arr, curr_ts, loss_leading_order)
        if algo == "pasting":
            if cost == "quadratic":
                Visualize_dyn_comp(time_lst[1:], [phi_stm[0,1:,:], phi_stm_leading_order[0,1:,:], phi_stm_ground_truth[0,1:,:]], curr_ts + "_training", "phi", [train_args["algo"], "leading_order", "ground_truth"])
                Visualize_dyn_comp(time_lst[1:], [phi_dot_stm[0,:,:], phi_dot_stm_leading_order[0,:,:], phi_dot_stm_ground_truth[0,:,:]], curr_ts + "_training", "phi_dot", [train_args["algo"], "leading_order", "ground_truth"])
            else:
                Visualize_dyn_comp(time_lst[1:], [phi_stm[0,1:,:], phi_stm_leading_order[0,1:,:]], curr_ts + "_training", "phi", [train_args["algo"], "leading_order"])
                Visualize_dyn_comp(time_lst[1:], [phi_dot_stm[0,:,:], phi_dot_stm_leading_order[0,:,:]], curr_ts + "_training", "phi_dot", [train_args["algo"], "leading_order"])
    else:
        curr_ts = "test"
    return model, loss_arr, prev_ts, curr_ts

def evaluation(dW_std, curr_ts, model = None, algo = "deep_hedging", cost = "quadratic", visualize_obs = 0, pasting_cutoff = 0, phi_0 = None, W_s0d = None, is_arr = False, pasting_T = None, time_lst = None, pasting_algo = "deep_hedging", report = False):
    assert algo in ["deep_hedging", "fbsde", "pasting", "leading_order", "ground_truth"]
    assert cost in ["quadratic", "power"]
    if cost == "quadratic":
        power = 2
    else:
        power = 3 / 2
    if time_lst is None:
        if pasting_T is None:
            time_lst = TIMESTAMPS[:(dW_std.shape[1] + 1)]
        else:
            time_lst = get_ts(pasting_cutoff, pasting_T)
    dynamic_factory = DynamicsFactory(time_lst, dW_std, W_s0d, g_dir = f"{drive_dir}/eva.txt")
    W_std, mu_stm, sigma_stmd, s_tm, xi_dd, lam_mm, alpha_md, beta_m = dynamic_factory.get_constant_processes()
    if model is not None:
        model.eval()
    if pasting_T is not None:
        dt = (T - pasting_cutoff) / T * TR / pasting_T
    else:
        dt = TR / T
    if algo == "deep_hedging":
        phi_dot_stm, phi_stm = dynamic_factory.deep_hedging(model, phi_0 = phi_0, dt = dt, orig=True)
    elif algo == "fbsde":
        if cost == "quadratic":
            phi_dot_stm, phi_stm, _ = dynamic_factory.fbsde_quad(model, dt = dt)
        else:
            phi_dot_stm, phi_stm, _ = dynamic_factory.fbsde_power(model, dt = dt)
        ### to match the dim
#        phi_dot_stm = phi_dot_stm[:,1:,:]
    elif algo == "pasting":
        phi_dot_stm, phi_stm = dynamic_factory.pasting(model, pasting_cutoff, dt = dt, pasting_T = pasting_T, pasting_algo = pasting_algo)
    elif algo == "leading_order":
        if cost == "quadratic":
            phi_dot_stm, phi_stm = dynamic_factory.leading_order_quad(model, phi_0 = phi_0, dt = dt)
        else:
            phi_dot_stm, phi_stm = dynamic_factory.leading_order_power(POWER, model, dt = dt)
    else:
        assert cost == "quadratic"
        phi_dot_stm, phi_stm = dynamic_factory.ground_truth(model, phi_0 = phi_0, dt = dt)
    loss_factory = LossFactory(time_lst, dW_std, W_s0d)
#    print(phi_dot_stm.shape, phi_stm.shape)
    if not report:
        loss = loss_factory.utility_loss(phi_dot_stm, phi_stm, power, is_arr = is_arr)
        if not is_arr:
            loss = float(loss.data)
        loss = loss * COEF_
    else:
#        mu, std, term_err = loss_factory.utility_loss_report(phi_dot_stm, phi_stm, power)
#        print(mu, std, term_err)
#        loss_msg = "mu = {:.3e}, std = {:.3e}; term_err = [".format(mu * COEF_, std * COEF_)
#        for m in range(len(term_err)):
#            loss_msg += "{:.3e}\t".format(term_err[m])
#        loss_msg = loss_msg.strip() + "]"
#        loss = loss_msg
        
        m1, m2, term_sum = loss_factory.utility_loss_report2(phi_dot_stm, phi_stm, power)
        loss = (m1, m2, term_sum)
    
#    Visualize_dyn(TIMESTAMPS[1:], phi_stm[0,1:,:], curr_ts, "phi")
#    Visualize_dyn(TIMESTAMPS[1:], phi_dot_stm[0,:,:], curr_ts, "phi_dot")
#    Visualize_dyn(TIMESTAMPS[1:], sigma_stmd, curr_ts, "sigma")
#    Visualize_dyn(TIMESTAMPS[1:], mu_stm, curr_ts, "mu")
#    Visualize_dyn(TIMESTAMPS[1:], s_tm, curr_ts, "s")
    phi_dot_stm = phi_dot_stm * COEF_
    phi_stm = phi_stm * COEF_
    return phi_dot_stm, phi_stm, loss

def get_dW(pasting_cutoff, pasting_T, n_sample, dW_orig = dW_STD, seed = None):
    dt = (T - pasting_cutoff) / T * TR / pasting_T
    if seed is not None:
        torch.manual_seed(seed)
    multi_normal = MultivariateNormal(torch.zeros((n_sample, pasting_T, N_BM)), BM_COV * dt)
    dW_STD_curr = multi_normal.sample().to(device = DEVICE)
    dW_STD_transfer = torch.cat((dW_orig[:,:pasting_cutoff,:], dW_STD_curr), dim = 1)
    return dW_STD_transfer

def get_ts(pasting_cutoff, pasting_T):
    dt = (T - pasting_cutoff) / T * TR / pasting_T
    ts_fst = TIMESTAMPS[:(pasting_cutoff + 1)]
    ts_sec = [float(ts_fst[-1]) + dt * (i + 1) for i in range(pasting_T)]
    return np.array(list(ts_fst) + list(ts_sec))

def transfer_learning(train_args, N_rounds = 5, n_train = 1, n_sample_lst = [128, 128], lr_lst = [1e-3, 1e-4], epoch_lst = [20000, 20000], retrain = False, report = True, seed = 0):
    if POWER == 2:
        cost = "quadratic"
    else:
        cost = "power"
    
    final_ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
    
    log_dir = None
    init_ts = None
    curr_ts = None
    log_dir = f"{drive_dir}/transfer_log_{final_ts}.txt"
    for i in range(N_rounds):
#        print(f"Round #{i+1}:")
#        with open(log_dir, "a") as f:
#            f.write(f"Round #{i+1}/{N_rounds} - Cutoff: {train_args['pasting_cutoff']}/{T}\n")
        
        ## Reset dW based on pasting_T
        ## TODO: MODIFY IT!!!
#        dW_STD_transfer = get_dW(train_args["pasting_cutoff"], train_args["pasting_T"], N_SAMPLE, dW_orig = dW_STD, seed = 0)
        dW_STD_transfer = dW_STD
        
        phi_dot_stm_leading_order, phi_stm_leading_order, loss_eval_leading_order = evaluation(dW_STD_transfer, final_ts, None, algo = "leading_order", cost = train_args["cost"], visualize_obs = 0, is_arr = True, pasting_cutoff = train_args["pasting_cutoff"], pasting_T = train_args["pasting_T"])
        if cost == "quadratic":
            phi_dot_stm_ground_truth, phi_stm_ground_truth, loss_eval_ground_truth = evaluation(dW_STD_transfer, final_ts, None, algo = "ground_truth", cost = train_args["cost"], visualize_obs = 0, pasting_cutoff = train_args["pasting_cutoff"], pasting_T = train_args["pasting_T"])
            
        ## Begin Training
        for j in range(n_train):
            if len(n_sample_lst) > 0:
                train_args["n_sample"] = n_sample_lst[min(j, len(n_sample_lst) - 1)]
            if len(lr_lst) > 0:
                train_args["lr"] = lr_lst[min(j, len(lr_lst) - 1)]
            if len(epoch_lst) > 0:
                train_args["epoch"] = epoch_lst[min(j, len(epoch_lst) - 1)]
            if retrain:
                model, loss_arr, prev_ts, curr_ts = training_pipeline(**train_args)
                train_args["retrain"] = False
                if i == 0 and j == 0:
                    init_ts = prev_ts
            else:
                pasting_algo = train_args["pasting_algo"]
                if train_args["pasting_algo"] == "deep_hedging":
                    if HIGH_DIM:
                        if POWER == 2:
                            model_name = f"{pasting_algo}_highdim_quad"
                        else:
                            model_name = f"{pasting_algo}_highdim_power_dt=0.5"
                    else:
                        if POWER == 2:
                            model_name = f"{pasting_algo}_quad_100"
                            model_name_dh = f"xdr_quad_252_1"
                        else:
                            model_name = f"{pasting_algo}_power_100"
                            model_name_dh = f"xdr_power_252_1"
                else:
                    if POWER == 2:
                        model_name = f"{pasting_algo}_quad_42"
                    else:
                        model_name = f"{pasting_algo}_power_42"
                model = torch.load(f"{drive_dir}/Models/{model_name}.pt")
                model_dh = torch.load(f"{drive_dir}/Models/{model_name_dh}.pt")
                model = model.get_pasting_frac(train_args["pasting_T"])
                model_dh.eval()
                prev_ts = None
                curr_ts = datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d-%H-%M-%S")
                loss_arr = [0]
            model.eval()
            phi_dot_stm_algo, phi_stm_algo, loss_eval_algo = evaluation(dW_STD_transfer, curr_ts, model, algo = train_args["algo"], cost = train_args["cost"], visualize_obs = 0, pasting_cutoff = train_args["pasting_cutoff"], is_arr = True, pasting_T = train_args["pasting_T"])
        
#            ## Write Logs
#            with open(log_dir, "a") as f:
#                f.write(f"\tIteration #{j+1}/{n_train} - Utility Loss: {float(torch.sum(loss_eval_algo))}\n")
        
        ## Modify pasting cutoff
        loss_diff = np.cumsum((loss_eval_algo - loss_eval_leading_order).cpu().data.numpy()[::-1])[::-1]
#        loss_diff = (loss_eval_algo - loss_eval_leading_order).cpu().data.numpy()
        loss_idx = loss_diff < 0
        ## TODO: MODIFY IT!!!
        pos = int(np.argmin(loss_diff))
        # Method 2
#        pos = len(loss_idx) - 1
#        while loss_idx[pos] and pos >= train_args["pasting_cutoff"]:
#            pos -= 1
        # Method 1
#        pos = int(np.argmax(loss_idx[train_args["pasting_cutoff"]:]))
        #pos = max(pos, train_args["pasting_cutoff"] - 1)
#        ts_lst = get_ts(train_args["pasting_cutoff"], train_args["pasting_T"])
#        ts_pos = ts_lst[pos] #ts_lst[(train_args["pasting_cutoff"] + 1):][pos]
#        pos = int(np.argmin(np.abs(ts_lst - ts_pos)) - 1 - train_args["pasting_cutoff"])
        pos = pos - train_args["pasting_cutoff"]
        #if pos == 0 and loss_idx[train_args["pasting_cutoff"]] == 0:
        if pos < 0:
            #print("Not Enough Training!")
            pos = train_args["pasting_cutoff"] + 0#pos
#            assert False
            #pos = max(0, T - (T - train_args["pasting_cutoff"]) * 2)
        else:
            pos = train_args["pasting_cutoff"] + pos
        if i < N_rounds - 1:
            train_args["pasting_T"] = int((T - pos) / (T - train_args["pasting_cutoff"]) * train_args["pasting_T"])
            train_args["pasting_cutoff"] = pos
            train_args["retrain"] = True
    
    torch.manual_seed(seed)

    REPEAT = 1#100
    m1_sum_leading_order = 0
    m2_sum_leading_order = 0
    term_sum_leading_order = 0
    m1_sum_ground_truth = 0
    m2_sum_ground_truth = 0
    term_sum_ground_truth = 0
    m1_sum_deep_hedging = 0
    m2_sum_deep_hedging = 0
    term_sum_deep_hedging = 0
    m1_sum_algo = 0
    m2_sum_algo = 0
    term_sum_algo = 0
    for i in range(REPEAT):
        if i % 1 == 0:
            dW_std = torch.randn([N_SAMPLE,N_BM,T])*np.sqrt(TR/T)
            dW_std = dW_std[:,0,:].reshape((N_SAMPLE, T, N_BM))
            dW_STD_transfer = dW_std
        else:
            dW_STD_transfer = -dW_STD_transfer

        phi_dot_stm_leading_order, phi_stm_leading_order, loss_eval_leading_order = evaluation(dW_STD_transfer, final_ts, None, algo = "leading_order", cost = train_args["cost"], visualize_obs = 0, is_arr = True, pasting_cutoff = train_args["pasting_cutoff"], pasting_T = train_args["pasting_T"], report = report)
        m1_sum_leading_order += loss_eval_leading_order[0]
        m2_sum_leading_order += loss_eval_leading_order[1]
        term_sum_leading_order += loss_eval_leading_order[2]
#        with open("seeds.txt", "a") as f:
#            f.write(f"Seed = {i + 1}, Utility = {round(loss_eval_leading_order[0] * COEF_ / 1e9, 2)}\n")
        if cost == "quadratic":
            phi_dot_stm_ground_truth, phi_stm_ground_truth, loss_eval_ground_truth = evaluation(dW_STD_transfer, final_ts, None, algo = "ground_truth", cost = train_args["cost"], visualize_obs = 0, pasting_cutoff = train_args["pasting_cutoff"], pasting_T = train_args["pasting_T"], report = report)
            m1_sum_ground_truth += loss_eval_ground_truth[0]
            m2_sum_ground_truth += loss_eval_ground_truth[1]
            term_sum_ground_truth += loss_eval_ground_truth[2]
        phi_dot_stm_deep_hedging, phi_stm_deep_hedging, loss_eval_deep_hedging = evaluation(dW_STD_transfer, curr_ts, model_dh, algo = "deep_hedging", cost = train_args["cost"], visualize_obs = 0, pasting_cutoff = train_args["pasting_cutoff"], is_arr = False, pasting_T = train_args["pasting_T"], report = report)
        m1_sum_deep_hedging += loss_eval_deep_hedging[0]
        m2_sum_deep_hedging += loss_eval_deep_hedging[1]
        term_sum_deep_hedging += loss_eval_deep_hedging[2]
        phi_dot_stm_algo, phi_stm_algo, loss_eval_algo = evaluation(dW_STD_transfer, curr_ts, model, algo = train_args["algo"], cost = train_args["cost"], visualize_obs = 0, pasting_cutoff = train_args["pasting_cutoff"], is_arr = False, pasting_T = train_args["pasting_T"], report = report)
        m1_sum_algo += loss_eval_algo[0]
        m2_sum_algo += loss_eval_algo[1]
        term_sum_algo += loss_eval_algo[2]
    m1_sum_leading_order /= REPEAT
    m2_sum_leading_order /= REPEAT
    term_sum_leading_order /= REPEAT
    loss_eval_leading_order = "mu = {:.3e}, std = {:.3e}; term_err = {:.3e}\t".format(m1_sum_leading_order * COEF_, np.sqrt(m2_sum_leading_order) * COEF_, term_sum_leading_order)
    if cost == "quadratic":
        m1_sum_ground_truth /= REPEAT
        m2_sum_ground_truth /= REPEAT
        term_sum_ground_truth /= REPEAT
        loss_eval_ground_truth = "mu = {:.3e}, std = {:.3e}; term_err = {:.3e}\t".format(m1_sum_ground_truth * COEF_, np.sqrt(m2_sum_ground_truth) * COEF_, term_sum_ground_truth)
    m1_sum_deep_hedging /= REPEAT
    m2_sum_deep_hedging /= REPEAT
    term_sum_deep_hedging /= REPEAT
    loss_eval_deep_hedging = "mu = {:.3e}, std = {:.3e}; term_err = {:.3e}\t".format(m1_sum_deep_hedging * COEF_, np.sqrt(m2_sum_deep_hedging) * COEF_, term_sum_deep_hedging)
    m1_sum_algo /= REPEAT
    m2_sum_algo /= REPEAT
    term_sum_algo /= REPEAT
    loss_eval_algo = "mu = {:.3e}, std = {:.3e}; term_err = {:.3e}\t".format(m1_sum_algo * COEF_, np.sqrt(m2_sum_algo) * COEF_, term_sum_algo)

    ## Write Logs
    with open("seeds2.txt", "a") as f:
        res = m1_sum_algo * COEF_ / 1e9
        f.write(f"Seed = {seed}, loss = {res}\n")
    
    train_args["transfer_learning"] = True
    return model, loss_arr, init_ts, curr_ts, train_args, model_dh

if POWER == 2:
    cost = "quadratic"
else:
    cost = "power"

## TODO: Adjust the arguments for training
train_args = {
    "algo": "pasting",
    "cost": cost,
    "model_name": "discretized_feedforward",
    "solver": "Adam",
    "hidden_lst": [10, 10, 10],#[50, 50, 50],
    "lr": 1e-3,
    "epoch": 20000,
    "train_freq": 1,
    "train_cut": 10,
    "decay": 0.1,
    "scheduler_step": 200000,
    "retrain": False,
    "pasting_cutoff": 202,#2470,#4990,#454,#908,#9880,#4940, #404,#,#2510, #4800,
    "pasting_T": 50,#200,#100,#40, #160, # None
    "pasting_algo": "deep_hedging",
    "n_sample": N_SAMPLE
}

#curr_ts = "test"

for seed in tqdm(range(100)):
    model, loss_arr, prev_ts, curr_ts, train_args, model_dh = transfer_learning(train_args, N_rounds = 1, n_train = 1, n_sample_lst = [128, 500], lr_lst = [1e-3, 1e-4], epoch_lst = [10, 10], seed = seed)
    model.eval()
