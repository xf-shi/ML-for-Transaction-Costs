# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from scipy.integrate import solve_ivp
from tqdm import tqdm
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
torch.manual_seed(1)
import pandas as pd
from sample_code_Deep_Q import *
from sample_code_FBSDE import *

# Market Variables
q=2.0
S_OUTSTANDING = 245714618646 # Total share outstanding
TIME          = 21 # Trading Horizon
TIME_STEP     = 168 # Discretization step
DT            = TIME/TIME_STEP 
GAMMA_BAR     = 8.30864*1e-14 # Aggregate risk aversion
KAPPA         = 1.
GAMMA_1       = GAMMA_BAR*(KAPPA+1)/KAPPA # Absolute risk aversion for both agents
GAMMA_2       = GAMMA_BAR*(KAPPA+1)
GAMMA_HAT     = (GAMMA_1-GAMMA_2)/(GAMMA_1+GAMMA_2)
GAMMA         = 0.5*(GAMMA_1+GAMMA_2)
XI_1          =  2.19*1e10 # Endowment volatility
XI_2          = -XI_1 #-XI_1
XI            = XI_1 
PHI_INITIAL   = S_OUTSTANDING*KAPPA/(KAPPA+1) # Initial allocation
ALPHA         = 1.8788381 # Frictionless volatility
ALPHA2        = ALPHA**2
MU_BAR        =  0.5*GAMMA*S_OUTSTANDING*ALPHA**2 # Frictionless return 
LAM           = 1.08e-10 # Transaction penalty

path_Q=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/DeepQ_trading{}_cost{}/".format(TIME,q)
isExist = os.path.exists(path_Q)
if not isExist:  
  os.makedirs(path_Q)
path_FBSDE=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/FBSDE_trading{}_cost{}/".format(TIME,q)
isExist = os.path.exists(path_FBSDE)
if not isExist:  
  os.makedirs(path_FBSDE)

# Batch size
N_SAMPLES = 300
# Dimension of Brownian Motion
BM_DIM    = 1
# Dimension of Forward Variables
FOR_DIM   = 1 
# Number of Backward Variables
BACK_DIM  = 2
# Training hyperparameter 
LR         = 1e-3
EPOCHS     = 1000

test = FBSDESolver(
                path_FBSDE=path_FBSDE,
                dt        = DT,
                 T         =TIME,
                 time_step = TIME_STEP,
                 n_samples = N_SAMPLES,
                 for_dim   = FOR_DIM,
                 back_dim  = BACK_DIM,
                 bm_dim    = BM_DIM,
                 s         = S_OUTSTANDING,
                 mu_bar    = MU_BAR,
                 alpha     = ALPHA,
                 gamma_bar = GAMMA_BAR,
                 gamma     = GAMMA,
                 gamma_hat = GAMMA_HAT,
                 xi        = XI,
                 lam       = LAM,
                 lam_dyn   = False,
                 q=q)
test.train(DECAY_LR=1.,epochs=EPOCHS ,path_size=N_SAMPLES,bm_dim= BM_DIM,time_step= TIME_STEP,dt = DT)
#test.restore_system()

"""# Deep Q-learning"""
LR_Utility = 1e-2 
N_SAMPLE_Utility=6000
TIME_STEP_U=TIME_STEP
result_Utility=TRAIN_Utility(train_on_gpu,path_Q,XI,PHI_INITIAL,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME,EPOCH=10,N=N_SAMPLE_Utility,T=TIME_STEP_U,HIDDEN_DIM_Utility=[10,15,10],loading=False,
      LR_Utility=LR_Utility,OPT_Utility="ADAM",
      saving=[10]) 
model_list_Utility=result_Utility['model_list']
loss_arr_Utility=result_Utility['loss']

