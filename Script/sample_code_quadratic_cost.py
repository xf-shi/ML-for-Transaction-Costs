import os
import numpy as np
import matplotlib.pyplot as plt
import torch
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

# TEST
def TEST(dW,model_list_Utility,test_samples,TEST_SEED=4):
    #dW: in the shape of SAMPLE_SIZE,bm_dim,TIME_STEP
    #test_samples: SAMPLE_SIZE
    #simulate the strategy of phi and phi_dot based on the model_list_Utility, Ground Truth, and the leading order approximation
    torch.manual_seed(TEST_SEED)
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
    for t in tqdm(range(T)):      
      PHI_dot_TRUTH[:,t]=-SQRT_GAMALP2_LAM*(PHI_TRUTH[:,t]-MU_BAR/GAMMA/ALPHA**2+XI/ALPHA*W[:,t].cpu().numpy())\
                                            *np.tanh(SQRT_GAMALP2_LAM*(T-t)/T*TIME)
      
      INTEGRAL_dWu[:,t+1]=INTEGRAL_dWu[:,t]\
                          +dW[:,0,t].cpu().numpy().reshape((-1,))/ np.cosh(SQRT_GAMALP2_LAM*(T-t)/T*TIME)                        
      PHI_TRUTH[:,t+1]=PHI_0.cpu().numpy().reshape((-1,))\
                      +-XI/ALPHA*W[:,t+1].cpu().numpy()\
                      +XI/ALPHA*np.cosh(SQRT_GAMALP2_LAM*(t-T)/T*TIME)*INTEGRAL_dWu[:,t+1]
    ###Leading Order
    PHI_dot_APP=np.zeros((test_samples,T))
    PHI_APP= np.zeros((test_samples,T+1))
    APP_INTEGRAL_dWu=np.zeros((test_samples,T+1))
    PHI_APP[:,0]=PHI_APP[:,0]+PHI_0.cpu().numpy().reshape((-1,))
    SQRT_GAMALP2_LAM=np.sqrt(GAMMA*ALPHA**2/LAM)
    for t in tqdm(range(T)):
      PHI_dot_APP[:,t]=-SQRT_GAMALP2_LAM*(PHI_APP[:,t]-MU_BAR/GAMMA/ALPHA**2+XI/ALPHA*W[:,t].cpu().numpy())      
      APP_INTEGRAL_dWu[:,t+1]=APP_INTEGRAL_dWu[:,t]\
                          +dW[:,0,t].cpu().numpy().reshape((-1,)) * np.exp(SQRT_GAMALP2_LAM*t/T*TIME)                       
      PHI_APP[:,t+1]=PHI_0.cpu().numpy().reshape((-1,))\
                      +-XI/ALPHA*W[:,t+1].cpu().numpy()\
                      +XI/ALPHA*np.exp(-SQRT_GAMALP2_LAM*t/T*TIME) *APP_INTEGRAL_dWu[:,t+1]
    result={
        "T":T,
        "Sample_XI_on_s":XI_W_on_s,
        "PHI_dot_on_s_Utility":PHI_dot_on_s,
        "PHI_dot_APP":PHI_dot_APP,
        "PHI_on_s_Utility":PHI_on_s,
        "PHI_APP":PHI_APP,
        "PHI_TRUTH":PHI_TRUTH,
        "PHI_dot_TRUTH":PHI_dot_TRUTH
        }
    return(result)

test_samples=30 #TEST SAMPLE SIZE
TARGET_test = torch.zeros(test_samples).reshape((test_samples, 1))
torch.manual_seed(1)
dW_test = train_data(n_samples=test_samples,bm_dim= BM_DIM,time_step= TIME_STEP,dt = DT)
dW_test_FBSDE=dW_test
W_test_FBSDE=torch.cumsum(dW_test_FBSDE[:,0,:], dim=1) 
W_test_FBSDE=torch.cat((torch.zeros((test_samples,1)),W_test_FBSDE),dim=1) 
XI_test_on_s_FBSDE = XI* W_test_FBSDE /S_OUTSTANDING
if train_on_gpu:
    XI_test_on_s_FBSDE = XI_test_on_s_FBSDE.to(device="cuda")
Test_result=TEST(dW_test,model_list_Q,test_samples)
T=Test_result["T"]
XI_test_on_s=Test_result["Sample_XI_on_s"]
PHI_dot_on_s_Utility=Test_result["PHI_dot_on_s_Utility"]
PHI_dot_APP=Test_result["PHI_dot_APP"]
PHI_on_s_Utility=Test_result["PHI_on_s_Utility"]
PHI_APP=Test_result["PHI_APP"]
PHI_TRUTH=Test_result["PHI_TRUTH"]
PHI_dot_TRUTH=Test_result["PHI_dot_TRUTH"]
test.system.sample_phi(dW_test_FBSDE)
PHI_dot_FBSDE=(test.system.D_Delta_t_value*XI-XI/ALPHA*dW_test_FBSDE[:,0,:].cpu().numpy())/DT
PHI_FBSDE=test.system.Delta_t_value*XI+MU_BAR/GAMMA/ALPHA/ALPHA-(S_OUTSTANDING*XI_test_on_s_FBSDE/ALPHA).cpu().numpy()

pathid=1 
fig = plt.figure(figsize=(10,4))
time   = np.linspace(0, TIME_STEP*DT, TIME_STEP+1)
time_FBSDE   = np.linspace(0, TIME_STEP*DT, TIME_STEP+1)
plt.subplot(1,2,1)
plt.plot(time_FBSDE[1:],PHI_dot_FBSDE[pathid,:], label = "FBSDE")
plt.plot(time[1:],S_OUTSTANDING*PHI_dot_on_s_Utility[pathid,:].cpu().detach().numpy(), label = "Deep Q-learning")
plt.plot(time[1:],PHI_dot_APP[pathid,:], label = "Leading-order")
plt.plot(time[1:],PHI_dot_TRUTH[pathid,:], label = "Ground Truth")
plt.hlines(0,xmin=0,xmax=TIME,linestyles='dotted')
plt.title("{}".format("$\\dot{\\varphi_t}$"))
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(time_FBSDE,PHI_FBSDE[pathid,:], label = "FBSDE")
plt.plot(time,S_OUTSTANDING*PHI_on_s_Utility[pathid,:].cpu().detach().numpy(), label = "Deep Q-learning")
plt.plot(time,PHI_APP[pathid,:], label = "Leading-order")
plt.plot(time,PHI_TRUTH[pathid,:], label = "Ground Truth")
plt.title("{}".format("$\\varphi_t$"))
plt.grid()
plt.legend()
fig.savefig(path+"trading{}_cost{}.png".format(TIME,q),bbox_inches='tight')


### UTILITY
FBSDEloss_trainbyUtility=criterion(PHI_dot_on_s_Utility.cpu()[:,-1],TARGET_test.reshape((-1,)))

Utilityloss_trainbyUtility = S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s,PHI_on_s_Utility,PHI_dot_on_s_Utility,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_trainbyUtility =S_OUTSTANDING* SD_Utility_on_s(XI_test_on_s,PHI_on_s_Utility,PHI_dot_on_s_Utility,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

### FBSDE
FBSDEloss_trainbyFBSDE=criterion(torch.from_numpy(PHI_dot_FBSDE)[:,-1]/S_OUTSTANDING,TARGET_test.reshape((-1,)))

Utilityloss_trainbyFBSDE = S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s_FBSDE.cpu(),torch.from_numpy(PHI_FBSDE)/S_OUTSTANDING,
                       torch.from_numpy(PHI_dot_FBSDE)/S_OUTSTANDING,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_trainbyFBSDE = S_OUTSTANDING*SD_Utility_on_s(XI_test_on_s_FBSDE.cpu(),torch.from_numpy(PHI_FBSDE)/S_OUTSTANDING,
                       torch.from_numpy(PHI_dot_FBSDE)/S_OUTSTANDING,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

### Leading Order
FBSDELoss_APP=criterion(torch.from_numpy(PHI_dot_APP)[:,-1]/S_OUTSTANDING, TARGET_test.reshape((-1,)))

UtilityLoss_APP=S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_APP)/S_OUTSTANDING,
                          torch.from_numpy(PHI_dot_APP)/S_OUTSTANDING,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_APP=S_OUTSTANDING*SD_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_APP)/S_OUTSTANDING,
                          torch.from_numpy(PHI_dot_APP)/S_OUTSTANDING,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

### Ground Truth
FBSDELoss_TRUTH=criterion(torch.from_numpy(PHI_dot_TRUTH)[:,-1]/S_OUTSTANDING, TARGET_test.reshape((-1,)))

UtilityLoss_TRUTH=S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_TRUTH)/S_OUTSTANDING,
                          torch.from_numpy(PHI_dot_TRUTH)/S_OUTSTANDING,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_TRUTH=S_OUTSTANDING*SD_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_TRUTH)/S_OUTSTANDING,
                          torch.from_numpy(PHI_dot_TRUTH)/S_OUTSTANDING,q,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

df=pd.DataFrame(columns=["Method",'E(Utility)',"sd(Utility)","MSE at T (on S)"])
df=df.append({"Method":"Utility Based","E(Utility)":-Utilityloss_trainbyUtility.data.cpu().numpy()/TIME,"sd(Utility)":"{:e}".format(UtilitySD_trainbyUtility.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDEloss_trainbyUtility.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"FBSDE","E(Utility)":-Utilityloss_trainbyFBSDE.data.cpu().numpy()/TIME,"sd(Utility)":"{:e}".format(UtilitySD_trainbyFBSDE.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDEloss_trainbyFBSDE.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"Leading Order","E(Utility)":-UtilityLoss_APP.data.cpu().numpy()/TIME,"sd(Utility)":"{:e}".format(UtilitySD_APP.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDELoss_APP.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"TRUTH","E(Utility)":-UtilityLoss_TRUTH.data.cpu().numpy()/TIME,"sd(Utility)":"{:e}".format(UtilitySD_TRUTH.data.cpu().numpy()/TIME),"MSE at T (on S)":FBSDELoss_TRUTH.data.cpu().numpy()},ignore_index=True)

df.to_csv(path+"trading{}_cost{}.csv".format(TIME,q), index=False, header=True)
