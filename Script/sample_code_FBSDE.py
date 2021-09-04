import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# simulate training dataset
def train_data(n_samples,bm_dim,time_step,dt):
    dW_samples = torch.randn([n_samples,bm_dim,time_step])*np.sqrt(dt)
    return dW_samples

# structure of volatility for a 1-dim backward component: a bm-dim vector process
class vol_t(torch.nn.Module):
    def __init__(self, n_input,n_hidden,n_output): #n_input=BM_DIM
        super(vol_t, self).__init__()
        self.layer1    = torch.nn.Linear(n_input, n_hidden)              # hidden layer
        self.bn1       = nn.BatchNorm1d(n_hidden, momentum=0.5)          # batch normalization
        # self.layer2    = torch.nn.Linear(n_hidden, n_hidden)
        # self.bn2       = nn.BatchNorm1d(n_hidden,momentum=0.5)
        self.outputZ   = torch.nn.Linear(n_hidden, n_output) # output layer
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))   
        # x = F.relu(self.bn2(self.layer2(x))) 
        x = self.outputZ(x)                                              # linear output
        return x

# Kaiming's initialization of parameters
def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('nn.Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

# all_zeros initialization of parameters
def weights_uniform_1(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('nn.Linear') != -1:
            # get the number of the inputs
            n = 1e4 #m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.fill_(0)
            m.bias.data.fill_(-3)

def weights_uniform_0(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('nn.Linear') != -1:
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)

class System(torch.nn.Module):
    def __init__(self, 
                path_FBSDE,
                 dt,
                 T,
                 time_step,
                 n_samples,
                 for_dim,
                 back_dim,
                 bm_dim,
                 s,
                 mu_bar,
                 alpha,
                 gamma_bar,
                 gamma,
                 gamma_hat,
                 xi,
                 lam,
                 lam_dyn,
                 q
                 ):
        super(System, self).__init__()

        ###
        self.end_vol=False
        self.D_Delta_t_value=np.zeros(0)
        self.Delta_t_value=np.zeros(0)
        self.q=q
        self.nu   = 0
        self.X_rate   = self.nu**2/2
        ###

        self.sigmaZ    = []
        self.sigmaY    = []
        self.T         = T
        self.dt        = dt
        self.time_step = time_step
        self.time      = np.linspace(0, self.T, self.time_step+1)
        self.n_samples = n_samples
        self.for_dim   = for_dim
        self.back_dim  = back_dim
        self.bm_dim    = bm_dim
        self.s         = s
        self.mu_bar    = mu_bar
        self.alpha     = alpha
        self.gamma     = gamma
        self.gamma_hat = gamma_hat
        self.gamma_bar = gamma_bar
        self.xi        = xi
        self.lam_level = lam
        self.lam_dyn   = lam_dyn
        self.model     = 'linear_state'
        
        self.save_dir = path_FBSDE + '/Equilibrium_dt_' + self.model + '_net_params.pkl'
        for t in range(self.time_step+1):
            sigmaZ_t = vol_t(n_input=2*bm_dim +2,n_hidden=15,n_output=bm_dim )
            sigmaY_t = vol_t(n_input=2*bm_dim +2,n_hidden=15,n_output=bm_dim )
            self.sigmaZ.append(sigmaZ_t)
            self.sigmaY.append(sigmaY_t)
        self.sigmaZ = nn.ModuleList(self.sigmaZ)
        self.sigmaY = nn.ModuleList(self.sigmaY)
    
    def forward(self, dW, lam_indepent=False):
        # Brownian motions starts at 0
        n_samples,bm_dim,time_step = dW.size()
        W_t    = torch.zeros(n_samples,bm_dim)*np.sqrt(self.T)
        
        # Initial value of forward variables
        Delta_t = torch.zeros(n_samples,1)
        X_t     = torch.ones(n_samples,bm_dim)/np.sqrt(bm_dim)
        XI_t    = torch.zeros(n_samples,1)

        # Input of neural network
        in_t  = torch.cat([W_t,X_t,XI_t,Delta_t], 1)
        
        # Initial guess of backward variables
        Z_t    = self.sigmaZ[0](in_t)
        Y_t    = torch.zeros_like(self.sigmaZ[0](in_t))

        c_lam = np.power(self.lam_level,1/(self.q-1))*self.xi
        c_z   = self.gamma*self.xi
        c_y   = self.gamma_bar*self.s

        for t in range(time_step):
            sigmaZ_t  = self.sigmaZ[t+1](in_t)
            dW_t      = dW[:,:,t].reshape(n_samples,bm_dim)
            dW_t_mat  = dW_t[:,0].reshape(n_samples,1)
            sigma_t   = torch.zeros_like(self.gamma_hat*sigmaZ_t*c_z)
            sigmaxi_t = torch.sqrt(torch.sum(torch.mul(X_t,X_t),dim=1).reshape(n_samples,1))

            # Dynamic of market
            if self.lam_dyn:    Lam_t = torch.sum(torch.mul(X_t,X_t),dim=1).reshape(n_samples,1)
            else:               Lam_t = torch.ones_like(sigmaZ_t)
            
            # Update rule
            drift_Z  = torch.mul(torch.mul(self.alpha+sigma_t,self.alpha+sigma_t),Delta_t)\
                        - torch.mul(torch.mul(sigma_t,self.alpha+sigma_t),XI_t)/self.alpha
            dZ_t     = c_z*(drift_Z*self.dt + torch.mul(sigmaZ_t,dW_t_mat))
            dXI_t    = torch.mul(sigmaxi_t,dW_t_mat)
            dX_t     = -self.X_rate*X_t*self.dt + self.nu*dW_t
            dDelta_t = torch.sign(Z_t)*torch.mul(torch.abs(Z_t),1/Lam_t).pow(1/(self.q-1))*self.dt/c_lam + dXI_t/self.alpha
            W_t      = W_t + dW_t
            Delta_t  = Delta_t + dDelta_t
            X_t      = X_t + dX_t
            XI_t     = XI_t + dXI_t
            Z_t      = Z_t + dZ_t
            in_t  = torch.cat([W_t,X_t,XI_t,Delta_t], 1)
        return Z_t
    
    def sample_phi(self,dW):
        n_samples,bm_dim,time_step = dW.size()
        W_t    = torch.zeros(n_samples,bm_dim)*np.sqrt(self.T)
        
        # Initial value of forward variables
        Delta_t = torch.zeros(n_samples,1)
        X_t     = torch.ones(n_samples,bm_dim)/np.sqrt(bm_dim)
        XI_t    = torch.zeros(n_samples,1)

        # Input of neural network
        in_t  = torch.cat([W_t,X_t,XI_t,Delta_t], 1)
        
        # Initial guess of backward variables
        Z_t    = self.sigmaZ[0](in_t)
        Y_t    = torch.zeros_like(self.sigmaZ[0](in_t))

        c_lam = np.power(self.lam_level,1/(self.q-1))*self.xi
        c_z   = self.gamma*self.xi
        c_y   = self.gamma_bar*self.s

        ###
        D_Delta_t_value=np.zeros((n_samples,self.time_step))
        Delta_t_value=np.zeros((n_samples,self.time_step+1))

        for t in range(time_step):
            sigmaZ_t  = self.sigmaZ[t+1](in_t)
            dW_t      = dW[:,:,t].reshape(n_samples,bm_dim)
            dW_t_mat  = dW_t[:,0].reshape(n_samples,1)
            sigma_t   = torch.zeros_like(self.gamma_hat*sigmaZ_t*c_z)
            sigmaxi_t = torch.sqrt(torch.sum(torch.mul(X_t,X_t),dim=1).reshape(n_samples,1))

            # Dynamic of market
            if self.lam_dyn:    Lam_t = torch.sum(torch.mul(X_t,X_t),dim=1).reshape(n_samples,1)
            else:               Lam_t = torch.ones_like(sigmaZ_t)
            
            # Update rule
            drift_Z  = torch.mul(torch.mul(self.alpha+sigma_t,self.alpha+sigma_t),Delta_t)\
                        - torch.mul(torch.mul(sigma_t,self.alpha+sigma_t),XI_t)/self.alpha
            dZ_t     = c_z*(drift_Z*self.dt + torch.mul(sigmaZ_t,dW_t_mat))
            dXI_t    = torch.mul(sigmaxi_t,dW_t_mat)
            dX_t     = -self.X_rate*X_t*self.dt + self.nu*dW_t
            dDelta_t = torch.sign(Z_t)*torch.mul(torch.abs(Z_t),1/Lam_t).pow(1/(self.q-1))*self.dt/c_lam + dXI_t/self.alpha
            W_t      = W_t + dW_t
            Delta_t  = Delta_t + dDelta_t
            X_t      = X_t + dX_t
            XI_t     = XI_t + dXI_t
            Z_t      = Z_t + dZ_t
            in_t  = torch.cat([W_t,X_t,XI_t,Delta_t], 1)
            D_Delta_t_value[:,t]=dDelta_t.detach().numpy().ravel()
            Delta_t_value[:,t+1]=Delta_t_value[:,t]+D_Delta_t_value[:,t]
        self.D_Delta_t_value=D_Delta_t_value
        self.Delta_t_value=Delta_t_value

class FBSDESolver(object):
    def __init__(self,
                 path_FBSDE,
                 dt,
                 T,
                 time_step,
                 n_samples,
                 for_dim,
                 back_dim,
                 bm_dim,
                 s,
                 mu_bar,
                 alpha,
                 gamma_bar,
                 gamma ,
                 gamma_hat,
                 xi,
                 lam ,
                 lam_dyn,
                 q,
                 LR=0.1):  
        ##
        self.system       = System(path_FBSDE,dt, T, time_step, n_samples, for_dim, back_dim, bm_dim, s, mu_bar, alpha, gamma_bar, gamma, gamma_hat, xi, lam, lam_dyn,q)
        self.optimizer    = torch.optim.Adam(self.system.parameters(), lr=LR, betas=(0.9, 0.99))
        self.scheduler    = StepLR(self.optimizer, step_size=1000, gamma=1)
        self.loss_func    = torch.nn.MSELoss()
        self.loss_history = []
    def train(self, epochs,DECAY_LR,path_size,bm_dim,time_step,dt):
        # asjust the learning rate by DECAY_LR
        for g in self.optimizer.param_groups:
                g['lr'] = g['lr']*DECAY_LR
        for epoch in tqdm(range(epochs)):
            dW_train       = train_data(n_samples=path_size,bm_dim=bm_dim,time_step=time_step,dt=dt)
            Z_T = self.system.forward(dW_train)
            loss     = self.loss_func(Z_T, torch.zeros_like(Z_T))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.data.numpy())
            self.scheduler.step()          
            if np.isinf(loss.data.cpu().numpy()) or np.isnan(loss.data.cpu().numpy()):
              print("\nFAIL")
              break
            if epoch % 1000 == 0:
                print('Epoch:', epoch,'LR',self.scheduler.get_last_lr(),'Loss:',loss)
                #dW = train_data(n_samples=2)
                self.save_system()    
    def save_system(self):
        torch.save(self.system.state_dict(), self.system.save_dir)
        print('done saving')
    def restore_system(self):
        self.system.load_state_dict(torch.load(self.system.save_dir))
        print('done loading')

