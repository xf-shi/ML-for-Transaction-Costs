import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
torch.manual_seed(1)
import pandas as pd
from joblib import Parallel, delayed
from sample_code_Deep_Hedging import *
from sample_code_FBSDE import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

path_Q=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/trading{}_cost{}/DeepQ/".format(TIME,q)
isExist = os.path.exists(path_Q)
if not isExist:  
  os.makedirs(path_Q)
path_FBSDE=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/trading{}_cost{}/FBSDE/".format(TIME,q)
isExist = os.path.exists(path_FBSDE)
if not isExist:  
  os.makedirs(path_FBSDE)
path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/trading{}_cost{}/".format(TIME,q)

# FBSDE
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
EPOCHS     = 10

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

# Deep Q-learning
EPOCH_Q=5
LR_Q = 1e-2 
N_SAMPLE_Q=300
TIME_STEP_Q=TIME_STEP
result_Q=TRAIN_Utility(train_on_gpu,path_Q,XI,PHI_INITIAL,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME,EPOCH=EPOCH_Q,n_samples=N_SAMPLE_Q,time_step=TIME_STEP_Q,HIDDEN_DIM_Utility=[10,15,10],loading=False,
      LR_Utility=LR_Q,OPT_Utility="ADAM",
      saving=[10]) 
model_list_Q=result_Q['model_list']
loss_arr_Q=result_Q['loss']


def TEST(dW,model_list_Utility,test_samples):
    with torch.no_grad():
        T=len(model_list_Utility)
        for model in model_list_Utility:
            model.eval()        
        PHI_0_on_s = torch.ones(test_samples)*PHI_INITIAL/S_OUTSTANDING
        PHI_0 = torch.ones(test_samples)*PHI_INITIAL
        DUMMY_1 = torch.ones(test_samples).reshape((test_samples, 1))
        if train_on_gpu:
            PHI_0_on_s = PHI_0_on_s.to(device="cuda")
            PHI_0 = PHI_0.to(device="cuda")
            DUMMY_1 = DUMMY_1.to(device="cuda")
        W=torch.cumsum(dW[:,0,:], dim=1) 
        W=torch.cat((torch.zeros((test_samples,1)),W),dim=1) 
        W_on_s=W/S_OUTSTANDING
        dW_on_s=dW/S_OUTSTANDING
        XI_W_on_s = XI* W /S_OUTSTANDING
        if train_on_gpu:
            XI_W_on_s = XI_W_on_s.to(device="cuda")
        PHI_on_s = torch.zeros((test_samples, T + 1))
        if train_on_gpu:
            PHI_on_s = PHI_on_s.to(device="cuda")
        PHI_on_s[:,0] = PHI_on_s[:,0]+PHI_0_on_s.reshape((-1,))
        PHI_dot_on_s = torch.zeros((test_samples, T ))
        if train_on_gpu:
            PHI_dot_on_s = PHI_dot_on_s.to(device="cuda")       
        for t in range(T):
            if train_on_gpu:
                t_tensor=t/T*TIME*torch.ones(test_samples).reshape(-1,1).cuda()            
                x_Utility=torch.cat((PHI_on_s[:,t].reshape(-1,1),XI_W_on_s[:,t].reshape(-1,1),t_tensor),dim=1).cuda()
            else: 
                t_tensor=t/T*TIME*torch.ones(test_samples).reshape(-1,1)
                x_Utility=torch.cat((PHI_on_s[:,t].reshape(-1,1),XI_W_on_s[:,t].reshape(-1,1),t_tensor),dim=1)        
            PHI_dot_on_s[:,t] = model_list_Utility[t](x_Utility).reshape(-1,)
            PHI_on_s[:,(t+1)] = PHI_on_s[:,t].reshape(-1)+PHI_dot_on_s[:,(t)].reshape(-1)*TIME/T
        for model in model_list_Utility:
            model.train() 
        ###Ground Truth
        PHI_dot_TRUTH=np.zeros((test_samples,T))
        PHI_TRUTH= np.zeros((test_samples,T+1))
        INTEGRAL_dWu=np.zeros((test_samples,T+1))
        PHI_TRUTH[:,0]=PHI_TRUTH[:,0]+PHI_0.cpu().numpy().reshape((-1,))
        SQRT_GAMALP2_LAM=np.sqrt(GAMMA*ALPHA**2/LAM)
        for t in range(T):          
          PHI_dot_TRUTH[:,t]=-SQRT_GAMALP2_LAM*(PHI_TRUTH[:,t]-MU_BAR/GAMMA/ALPHA**2+XI/ALPHA*W[:,t].cpu().numpy())\
                                                *np.tanh(SQRT_GAMALP2_LAM*(T-t)/T*TIME)
          
          INTEGRAL_dWu[:,t+1]=INTEGRAL_dWu[:,t]\
                              +dW[:,0,t].cpu().numpy().reshape((-1,))/ np.cosh(SQRT_GAMALP2_LAM*(T-t)/T*TIME)                        
          PHI_TRUTH[:,t+1]=PHI_0.cpu().numpy().reshape((-1,))\
                          +-XI/ALPHA*W[:,t+1].cpu().numpy()\
                          +XI/ALPHA*np.cosh(SQRT_GAMALP2_LAM*(t-T)/T*TIME)*INTEGRAL_dWu[:,t+1]                
        ###Leading Order
        PHI_dot_APP_on_s=np.zeros((test_samples,T))
        PHI_APP_on_s= np.zeros((test_samples,T+1))
        APP_INTEGRAL_dWu_on_s=np.zeros((test_samples,T+1))
        PHI_APP_on_s[:,0]=PHI_APP_on_s[:,0]+PHI_0_on_s.cpu().numpy().reshape((-1,))
        SQRT_GAMALP2_LAM=np.sqrt(GAMMA*ALPHA**2/LAM)
        for t in range(T):          
          PHI_dot_APP_on_s[:,t]=-SQRT_GAMALP2_LAM*(PHI_APP_on_s[:,t]-MU_BAR/GAMMA/ALPHA**2/S_OUTSTANDING+XI/ALPHA*W_on_s[:,t].cpu().numpy())             
          APP_INTEGRAL_dWu_on_s[:,t+1]=APP_INTEGRAL_dWu_on_s[:,t]\
                              +dW_on_s[:,0,t].cpu().numpy().reshape((-1,)) * np.exp(SQRT_GAMALP2_LAM*t/T*TIME)                       
          PHI_APP_on_s[:,t+1]=PHI_0_on_s.cpu().numpy().reshape((-1,))\
                          +-XI/ALPHA*W_on_s[:,t+1].cpu().numpy()\
                          +XI/ALPHA*np.exp(-SQRT_GAMALP2_LAM*t/T*TIME) *APP_INTEGRAL_dWu_on_s[:,t+1]
    result={
        "T":T,
        "Sample_XI_on_s":XI_W_on_s,
        "PHI_dot_on_s_Utility":PHI_dot_on_s,
        "PHI_dot_APP_on_s":PHI_dot_APP_on_s,
        "PHI_on_s_Utility":PHI_on_s,
        "PHI_APP_on_s":PHI_APP_on_s,
        "PHI_TRUTH":PHI_TRUTH,
        "PHI_dot_TRUTH":PHI_dot_TRUTH,
        "PHI_on_s_TRUTH":PHI_TRUTH/S_OUTSTANDING,
        "PHI_dot_on_s_TRUTH":PHI_dot_TRUTH/S_OUTSTANDING
        }
    return(result)

def big_test(test_samples,REPEAT,model_list_Utility=model_list_Q):
    TARGET_test = torch.zeros(test_samples).reshape((test_samples, 1))
    mu_Utility = 0.0
    mu2_Utility = 0.0
    FBSDELOSS_Utility = 0.0
    mu_FBSDE = 0.0
    mu2_FBSDE = 0.0
    FBSDELOSS_FBSDE = 0.0
    mu_LO = 0.0
    mu2_LO = 0.0
    FBSDELOSS_LO = 0.0
    mu_TRUTH = 0.0
    mu2_TRUTH = 0.0
    FBSDELOSS_TRUTH = 0.0
    for itr in tqdm(range(REPEAT)):
        dW_test = train_data(n_samples=test_samples,bm_dim= BM_DIM,time_step= TIME_STEP,dt = DT)
        dW_test_FBSDE=dW_test
        W_test_FBSDE=torch.cumsum(dW_test_FBSDE[:,0,:], dim=1) #ttttt
        W_test_FBSDE=torch.cat((torch.zeros((test_samples,1)),W_test_FBSDE),dim=1) 
        XI_test_on_s_FBSDE = XI* W_test_FBSDE /S_OUTSTANDING
        if train_on_gpu:
          XI_test_on_s_FBSDE = XI_test_on_s_FBSDE.to(device="cuda")
        Test_result=TEST(dW_test,model_list_Utility,test_samples)
        T=Test_result["T"]
        XI_test_on_s=Test_result["Sample_XI_on_s"]
        PHI_dot_on_s_Utility=Test_result["PHI_dot_on_s_Utility"]
        PHI_dot_APP_on_s=Test_result["PHI_dot_APP_on_s"]
        PHI_dot_TRUTH_on_s=Test_result["PHI_dot_on_s_TRUTH"]
        PHI_on_s_Utility=Test_result["PHI_on_s_Utility"]
        PHI_APP_on_s=Test_result["PHI_APP_on_s"]
        PHI_TRUTH_on_s=Test_result["PHI_on_s_TRUTH"]
        #FBSDE
        test.system.sample_phi(dW_test_FBSDE)
        PHI_dot_FBSDE=(test.system.D_Delta_t_value*XI-XI/ALPHA*dW_test_FBSDE[:,0,:].cpu().numpy())/DT
        PHI_dot_FBSDE_on_s=PHI_dot_FBSDE/S_OUTSTANDING
        PHI_FBSDE=test.system.Delta_t_value*XI+MU_BAR/GAMMA/ALPHA/ALPHA-(S_OUTSTANDING*XI_test_on_s_FBSDE/ALPHA).cpu().numpy()
        PHI_FBSDE_on_s=PHI_FBSDE/S_OUTSTANDING
        ### UTILITY
        FBSDEloss_trainbyUtility=criterion(PHI_dot_on_s_Utility.cpu()[:,-1],TARGET_test.reshape((-1,)))
        Utilityloss_trainbyUtility_on_s = Mean_Utility_on_s(XI_test_on_s.cpu(),PHI_on_s_Utility.cpu(),PHI_dot_on_s_Utility.cpu(),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        UtilitylossSQ_trainbyUtility_on_s = MeanSQ_Utility_on_s(XI_test_on_s.cpu(),PHI_on_s_Utility.cpu(),PHI_dot_on_s_Utility.cpu(),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        FBSDELOSS_Utility=FBSDELOSS_Utility +FBSDEloss_trainbyUtility
        mu_Utility =mu_Utility + Utilityloss_trainbyUtility_on_s
        mu2_Utility =mu2_Utility+ UtilitylossSQ_trainbyUtility_on_s 
        ### FBSDE
        FBSDEloss_trainbyFBSDE=criterion(torch.from_numpy(PHI_dot_FBSDE_on_s)[:,-1],TARGET_test.reshape((-1,)))
        Utilityloss_trainbyFBSDE_on_s = Mean_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_FBSDE_on_s),torch.from_numpy(PHI_dot_FBSDE_on_s),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        UtilitylossSQ_trainbyFBSDE_on_s =MeanSQ_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_FBSDE_on_s),torch.from_numpy(PHI_dot_FBSDE_on_s),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        FBSDELOSS_FBSDE=FBSDELOSS_FBSDE +FBSDEloss_trainbyFBSDE
        mu_FBSDE =mu_FBSDE+ Utilityloss_trainbyFBSDE_on_s
        mu2_FBSDE =mu2_FBSDE+ UtilitylossSQ_trainbyFBSDE_on_s 
        ### LO
        FBSDELoss_APP=criterion(torch.from_numpy(PHI_dot_APP_on_s)[:,-1], TARGET_test.reshape((-1,)))
        Utilityloss_LO_on_s = Mean_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_APP_on_s),torch.from_numpy(PHI_dot_APP_on_s),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        UtilitylossSQ_LO_on_s = MeanSQ_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_APP_on_s),torch.from_numpy(PHI_dot_APP_on_s),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        FBSDELOSS_LO=FBSDELOSS_LO +FBSDELoss_APP
        mu_LO =mu_LO+ Utilityloss_LO_on_s
        mu2_LO =mu2_LO+ UtilitylossSQ_LO_on_s 
        ### TRUTH
        FBSDELoss_byTRUTH=criterion(torch.from_numpy(PHI_dot_TRUTH_on_s)[:,-1], TARGET_test.reshape((-1,)))
        Utilityloss_TRUTH_on_s = Mean_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_TRUTH_on_s),torch.from_numpy(PHI_dot_TRUTH_on_s),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        UtilitylossSQ_TRUTH_on_s = MeanSQ_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_TRUTH_on_s),torch.from_numpy(PHI_dot_TRUTH_on_s),q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        FBSDELOSS_TRUTH=FBSDELOSS_TRUTH +FBSDELoss_byTRUTH
        mu_TRUTH =mu_TRUTH+ Utilityloss_TRUTH_on_s
        mu2_TRUTH =mu2_TRUTH+ UtilitylossSQ_TRUTH_on_s 
        if itr==0:
            ###plot
            pathid=1#000
            fig = plt.figure(figsize=(10,4))
            time   = np.linspace(0, TIME_STEP*DT, TIME_STEP+1)
            time_FBSDE   = np.linspace(0, TIME_STEP*DT, TIME_STEP+1)
            ax1=plt.subplot(1,2,1)
            ax1.plot(time_FBSDE[1:],PHI_dot_FBSDE[pathid,:], label = "FBSDE")
            ax1.plot(time[1:],S_OUTSTANDING*PHI_dot_on_s_Utility[pathid,:].cpu().detach().numpy(), label = "Deep Hedging")
            ax1.plot(time[1:],S_OUTSTANDING*PHI_dot_APP_on_s[pathid,:], label = "Leading-order")
            ax1.plot(time[1:],S_OUTSTANDING*PHI_dot_TRUTH_on_s[pathid,:], label = "Ground Truth")
            ax1.hlines(0,xmin=0,xmax=TIME,linestyles='dotted')
            ax1.title.set_text("{}".format("$\\dot{\\varphi_t}$"))
            ax1.grid()
            ax2=plt.subplot(1,2,2)
            ax2.plot(time_FBSDE,PHI_FBSDE[pathid,:], label = "FBSDE")
            ax2.plot(time,S_OUTSTANDING*PHI_on_s_Utility[pathid,:].cpu().detach().numpy(), label = "Deep Hedging")
            ax2.plot(time,S_OUTSTANDING*PHI_APP_on_s[pathid,:], label = "Leading-order")
            ax2.plot(time,S_OUTSTANDING*PHI_TRUTH_on_s[pathid,:], label = "Ground Truth")
            ax2.title.set_text("{}".format("${\\varphi_t}$"))
            ax2.grid()
            box2=ax2.get_position()
            ax2.legend(loc="lower left", bbox_to_anchor=(box2.width*3,box2.height))
            plt.savefig(path+"q{} trading{}.pdf".format(q,TIME), bbox_inches='tight')
    big_result={"mu_Utility":mu_Utility ,
      "mu2_Utility":mu2_Utility ,
      "FBSDELOSS_Utility":FBSDELOSS_Utility, 
      "mu_FBSDE":mu_FBSDE ,
      "mu2_FBSDE":mu2_FBSDE, 
      "FBSDELOSS_FBSDE":FBSDELOSS_FBSDE ,
      "mu_LO":mu_LO,
      "mu2_LO":mu2_LO, 
      "FBSDELOSS_LO":FBSDELOSS_LO,
      "mu_TRUTH":mu_TRUTH,
      "mu2_TRUTH":mu2_TRUTH, 
      "FBSDELOSS_TRUTH":FBSDELOSS_TRUTH
      }
    return big_result

test_size=3 #10000 later
REPEAT = 2 #10000
torch.manual_seed(1)
n_cpu=1 #15
batch_size = int(math.ceil(REPEAT / n_cpu))
big_result_arr = Parallel(n_jobs=n_cpu)(delayed(big_test)(
            test_size, min(REPEAT, batch_size * (i + 1)) - batch_size * i, model_list_Q
        ) for i in range(n_cpu))
big_result = big_result_arr[0]
for i in range(1, len(big_result_arr)):
    for key in big_result:
        big_result[key] += big_result_arr[i][key]
mu_Utility=big_result["mu_Utility"]
mu2_Utility =big_result["mu2_Utility"]
FBSDELOSS_Utility=big_result["FBSDELOSS_Utility"] 
mu_FBSDE=  big_result["mu_FBSDE"]
mu2_FBSDE=    big_result["mu2_FBSDE"] 
FBSDELOSS_FBSDE=    big_result["FBSDELOSS_FBSDE"]

mu_LO=    big_result["mu_LO"]
mu2_LO=    big_result["mu2_LO"] 
FBSDELOSS_LO=  big_result["FBSDELOSS_LO"]
mu_TRUTH=  big_result["mu_TRUTH"]
mu2_TRUTH=  big_result["mu2_TRUTH"] 
FBSDELOSS_TRUTH=  big_result["FBSDELOSS_TRUTH"]

FBSDELOSS_Utility=FBSDELOSS_Utility/ REPEAT
mu_Utility = mu_Utility/REPEAT
mu2_Utility = mu2_Utility/REPEAT
sigma_Utility = (mu2_Utility-mu_Utility**2)*(REPEAT*test_size)/(REPEAT*test_size-1)
sigma_Utility = sigma_Utility**0.5

FBSDELOSS_FBSDE=FBSDELOSS_FBSDE/ REPEAT
mu_FBSDE =mu_FBSDE/ REPEAT
mu2_FBSDE =mu2_FBSDE / REPEAT
sigma_FBSDE = (mu2_FBSDE-mu_FBSDE**2)*(REPEAT*test_size)/(REPEAT*test_size-1)
sigma_FBSDE = sigma_FBSDE **0.5

FBSDELOSS_LO=FBSDELOSS_LO/ REPEAT
mu_LO =mu_LO/ REPEAT
mu2_LO =mu2_LO/ REPEAT
sigma_LO = (mu2_LO-mu_LO**2)*(REPEAT*test_size)/(REPEAT*test_size-1)
sigma_LO=sigma_LO**0.5

FBSDELOSS_TRUTH=FBSDELOSS_TRUTH/ REPEAT
mu_TRUTH=mu_TRUTH/ REPEAT
mu2_TRUTH =mu2_TRUTH/ REPEAT
sigma_TRUTH = (mu2_TRUTH-mu_TRUTH**2)*(REPEAT*test_size)/(REPEAT*test_size-1)
sigma_TRUTH=sigma_TRUTH**0.5
#
mu_Utility = mu_Utility*S_OUTSTANDING
sigma_Utility=sigma_Utility*S_OUTSTANDING
mu_FBSDE = mu_FBSDE*S_OUTSTANDING
sigma_FBSDE=sigma_FBSDE*S_OUTSTANDING
mu_LO= mu_LO*S_OUTSTANDING
sigma_LO=sigma_LO*S_OUTSTANDING
mu_TRUTH= mu_TRUTH*S_OUTSTANDING
sigma_TRUTH=sigma_TRUTH*S_OUTSTANDING

df=pd.DataFrame(columns=["Method",'E(Utility)',"sd(Utility)","MSE at T (on S)"])
df=df.append({"Method":"Utility Based","E(Utility)":"{:e}".format(-mu_Utility.data.cpu().numpy()/TIME),"sd(Utility)":"{:e}".format(sigma_Utility.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDELOSS_Utility.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"FBSDE","E(Utility)":"{:e}".format(-mu_FBSDE.data.cpu().numpy()/TIME),"sd(Utility)":"{:e}".format(sigma_FBSDE.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDELOSS_FBSDE.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"Leading Order","E(Utility)":"{:e}".format(-mu_LO.data.cpu().numpy()/TIME),"sd(Utility)":"{:e}".format(sigma_LO.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDELOSS_LO.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"Ground Truth","E(Utility)":"{:e}".format(-mu_TRUTH.data.cpu().numpy()/TIME),"sd(Utility)":"{:e}".format(sigma_TRUTH.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDELOSS_TRUTH.data.cpu().numpy()},ignore_index=True)

df.to_csv(path+"trading{}_cost{}.csv".format(TIME,q), index=False, header=True)