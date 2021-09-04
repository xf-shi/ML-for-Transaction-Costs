import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
torch.manual_seed(1)
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
import pandas as pd
from sample_code_Deep_Q import *
from sample_code_FBSDE import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Market Variables
q=3/2
S_OUTSTANDING = 245714618646 # Total share outstanding
TIME          = 21 # Trading Horizon
TIME_STEP     = 168 # Discretization step
DT            = TIME/TIME_STEP 
GAMMA_BAR     = 8.30864e-14 # Aggregate risk aversion
KAPPA         = 1.
GAMMA_1       = GAMMA_BAR*(KAPPA+1)/KAPPA # Absolute risk aversion for both agents
GAMMA_2       = GAMMA_BAR*(KAPPA+1)
GAMMA_HAT     = (GAMMA_1-GAMMA_2)/(GAMMA_1+GAMMA_2)
GAMMA         = 0.5*(GAMMA_1+GAMMA_2)
XI_1          = 2.33e10 # Endowment volatility
XI_2          = -XI_1
XI            = XI_1 
PHI_INITIAL   = S_OUTSTANDING*KAPPA/(KAPPA+1) # Initial allocation
ALPHA         = 1.8788381 # Frictionless volatility
ALPHA2        = ALPHA**2
MU_BAR        =  0.5*GAMMA*S_OUTSTANDING*ALPHA**2 # Frictionless return
LAM           = 5.22e-6 # Transaction penalty

path_Q=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/trading{}_cost{}/DeepQ/".format(TIME,q)
isExist = os.path.exists(path_Q)
if not isExist:  
  os.makedirs(path_Q)
path_FBSDE=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/trading{}_cost{}/FBSDE/".format(TIME,q)
isExist = os.path.exists(path_FBSDE)
if not isExist:  
  os.makedirs(path_FBSDE)
path=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))+"/Data/trading{}_cost{}/".format(TIME,q) 

# FBSDE solver
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
                 dt     = DT,
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
result_Q=TRAIN_Utility(train_on_gpu,path_Q,XI,PHI_INITIAL,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME,EPOCH=EPOCH_Q,N=N_SAMPLE_Q,T=TIME_STEP_Q,HIDDEN_DIM_Utility=[10,15,10],loading=False,
      LR_Utility=LR_Q,OPT_Utility="ADAM",
      saving=[10]) 
model_list_Q=result_Q['model_list']
loss_arr_Q=result_Q['loss']

# TEST
# the leading order ODE is solved by Mathematica and the solution TABLE is stored in EVA.txt and X.txt
pathEVA='../Data/'
EVA=np.loadtxt(pathEVA+"EVA.txt") #value of g(x) of TABLE
XXX=np.loadtxt(pathEVA+"X.txt") #value of x of TABLE
EVA=np.vstack((np.linspace(0,50,EVA.shape[0]), EVA)).T
XM=50
def g_q_Mathematica(x):
    if np.abs(x)>XM:
        g_value= -np.sign(x)*q*(q-1)**(-(q-1)/q)*np.abs(x)**(2*(q-1)/q)           
    else:
        g_value=-np.sign(x)*np.abs(EVA[(np.abs(EVA[:,0] - np.abs(x))).argmin(),1])
    return g_value
g_q_Mathematica_vec=np.vectorize(g_q_Mathematica)

def TEST(dW,model_list_Utility,N,TEST_SEED=4):
    #dW: in the shape of SAMPLE_SIZE,bm_dim,TIME_STEP
    #N: SAMPLE_SIZE
    #simulate the strategy of phi and phi_dot based on the model_list_Utility, Ground Truth, and the leading order approximation
    torch.manual_seed(TEST_SEED)
    T=len(model_list_Utility)
    for model in model_list_Utility:
        model.eval()
    PHI_0_on_s = torch.ones(N)*PHI_INITIAL/S_OUTSTANDING
    PHI_0 = torch.ones(N)*PHI_INITIAL
    DUMMY_1 = torch.ones(N).reshape((N, 1))
    if train_on_gpu:
        PHI_0_on_s = PHI_0_on_s.to(device="cuda")
        PHI_0 = PHI_0.to(device="cuda")
        DUMMY_1 = DUMMY_1.to(device="cuda")
    W=torch.cumsum(dW[:,0,:], dim=1) #ttttt
    W=torch.cat((torch.zeros((N,1)),W),dim=1) 
    XI_W_on_s = XI* W /S_OUTSTANDING
    if train_on_gpu:
        XI_W_on_s = XI_W_on_s.to(device="cuda")
    PHI_on_s = torch.zeros((N, T + 1))
    if train_on_gpu:
        PHI_on_s = PHI_on_s.to(device="cuda")
    PHI_on_s[:,0] = PHI_on_s[:,0]+PHI_0_on_s.reshape((-1,))
    PHI_dot_on_s = torch.zeros((N, T ))
    if train_on_gpu:
        PHI_dot_on_s = PHI_dot_on_s.to(device="cuda")
    for t in range(T):
        if train_on_gpu:
            t_tensor=t/T*TIME*torch.ones(N).reshape(-1,1).cuda()            
            x_Utility=torch.cat((PHI_on_s[:,t].reshape(-1,1),XI_W_on_s[:,t].reshape(-1,1),t_tensor),dim=1).cuda()
        else: 
            t_tensor=t/T*TIME*torch.ones(N).reshape(-1,1)
            x_Utility=torch.cat((PHI_on_s[:,t].reshape(-1,1),XI_W_on_s[:,t].reshape(-1,1),t_tensor),dim=1)        
        PHI_dot_on_s[:,t] = model_list_Utility[t](x_Utility).reshape(-1,)
        PHI_on_s[:,(t+1)] = PHI_on_s[:,t].reshape(-1)+PHI_dot_on_s[:,(t)].reshape(-1)*TIME/T
    for model in model_list_Utility:
        model.train()    
    
    # Leading Order by Mathematica
    PHI_dot_APP_Mathematica = np.zeros((N,T))
    PHI_APP_Mathematica = np.zeros((N,T+1))
    PHI_APP_Mathematica[:,0] = PHI_APP_Mathematica[:,0]+PHI_0.cpu().numpy().reshape((-1,))
    for t in tqdm(range(T)):
        XIt=XI_W_on_s.cpu().numpy()[:,t] *S_OUTSTANDING
        PHIt=PHI_APP_Mathematica[:,t]
        PHIBARt=MU_BAR/GAMMA/ALPHA/ALPHA-XIt/ALPHA
        xxx=2**((q-1)/(q+2))*(q*GAMMA*ALPHA*ALPHA/LAM)**(1/(q+2))*(ALPHA/XI)**(2*q/(q+2))*(PHIt-PHIBARt)
        PHI_dot_APP_Mathematica[:,t]=(-np.sign(PHIt-PHIBARt)
                  *(q*GAMMA*XI**4/8/LAM/ALPHA/ALPHA)**(1/(q+2))
                  *abs(g_q_Mathematica_vec(xxx)/q)**(1/(q-1))
                  )        
        PHI_APP_Mathematica[:,t+1]=PHI_APP_Mathematica[:,t]+PHI_dot_APP_Mathematica[:,t]*TIME/T
    result={
        "T":T,
        "Sample_XI_on_s":XI_W_on_s,
        "PHI_dot_on_s_Utility":PHI_dot_on_s,
        "PHI_dot_APP_Mathematica":PHI_dot_APP_Mathematica,
        "PHI_on_s_Utility":PHI_on_s,
        "PHI_APP_Mathematica":PHI_APP_Mathematica,
        }
    return(result)

test_size=30 #TEST SAMPLE SIZE
TARGET_test = torch.zeros(test_size).reshape((test_size, 1))
torch.manual_seed(1)
dW_test = train_data(n_samples=test_size,bm_dim=BM_DIM,time_step=TIME_STEP,dt=DT)
dW_test_FBSDE=dW_test
W_test_FBSDE=torch.cumsum(dW_test_FBSDE[:,0,:], dim=1) #ttttt
W_test_FBSDE=torch.cat((torch.zeros((test_size,1)),W_test_FBSDE),dim=1) 
XI_test_on_s_FBSDE = XI* W_test_FBSDE /S_OUTSTANDING
if train_on_gpu:
    XI_test_on_s_FBSDE = XI_test_on_s_FBSDE.to(device="cuda")
Test_result=TEST(dW_test,model_list_Q,test_size)
T=Test_result["T"]
XI_test_on_s=Test_result["Sample_XI_on_s"]
PHI_dot_on_s_Utility=Test_result["PHI_dot_on_s_Utility"]
PHI_dot_APP_Mathematica=Test_result["PHI_dot_APP_Mathematica"]
PHI_on_s_Utility=Test_result["PHI_on_s_Utility"]
PHI_APP_Mathematica=Test_result["PHI_APP_Mathematica"]
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
plt.plot(time[1:],PHI_dot_APP_Mathematica[pathid,:], label = "Leading-order")
plt.hlines(0,xmin=0,xmax=TIME,linestyles='dotted')
plt.title("{}".format("$\\dot{\\varphi_t}$"))
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(time_FBSDE,PHI_FBSDE[pathid,:], label = "FBSDE")
plt.plot(time,S_OUTSTANDING*PHI_on_s_Utility[pathid,:].cpu().detach().numpy(), label = "Deep Q-learning")
plt.plot(time,PHI_APP_Mathematica[pathid,:], label = "Leading-order")
plt.title("{}".format("$\\varphi_t$"))
plt.grid()
plt.legend()
fig.savefig(path+"trading{}_cost{}.png".format(TIME,q),bbox_inches='tight')

### UTILITY
FBSDEloss_trainbyUtility=criterion(S_OUTSTANDING*PHI_dot_on_s_Utility.cpu()[:,-1],TARGET_test.reshape((-1,)))
FBSDEloss_trainbyUtility=criterion(PHI_dot_on_s_Utility.cpu()[:,-1],TARGET_test.reshape((-1,)))

Utilityloss_trainbyUtility = S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s,PHI_on_s_Utility,PHI_dot_on_s_Utility,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_trainbyUtility =S_OUTSTANDING* SD_Utility_on_s(XI_test_on_s,PHI_on_s_Utility,PHI_dot_on_s_Utility,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

### FBSDE
FBSDEloss_trainbyFBSDE=criterion(torch.from_numpy(PHI_dot_FBSDE)[:,-1]/S_OUTSTANDING,TARGET_test.reshape((-1,)))

Utilityloss_trainbyFBSDE = S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s_FBSDE.cpu(),torch.from_numpy(PHI_FBSDE)/S_OUTSTANDING,
                       torch.from_numpy(PHI_dot_FBSDE)/S_OUTSTANDING,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_trainbyFBSDE = S_OUTSTANDING*SD_Utility_on_s(XI_test_on_s_FBSDE.cpu(),torch.from_numpy(PHI_FBSDE)/S_OUTSTANDING,
                       torch.from_numpy(PHI_dot_FBSDE)/S_OUTSTANDING,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

### Leading Order
FBSDELoss_APP_Mathematica=criterion(torch.from_numpy(PHI_dot_APP_Mathematica)[:,-1], TARGET_test.reshape((-1,)))
FBSDELoss_APP_Mathematica=criterion(torch.from_numpy(PHI_dot_APP_Mathematica)[:,-1]/S_OUTSTANDING, TARGET_test.reshape((-1,)))

UtilityLoss_APP_Mathematica=S_OUTSTANDING*Mean_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_APP_Mathematica)/S_OUTSTANDING,
                          torch.from_numpy(PHI_dot_APP_Mathematica)/S_OUTSTANDING,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
UtilitySD_APP_Mathematica=S_OUTSTANDING*SD_Utility_on_s(XI_test_on_s.cpu(),torch.from_numpy(PHI_APP_Mathematica)/S_OUTSTANDING,
                          torch.from_numpy(PHI_dot_APP_Mathematica)/S_OUTSTANDING,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)

df=pd.DataFrame(columns=["Method",'E(Utility)',"sd(Utility)","MSE at T (on S)"])
df=df.append({"Method":"Utility Based","E(Utility)":-Utilityloss_trainbyUtility.data.cpu().numpy(),"sd(Utility)":"{:e}".format(UtilitySD_trainbyUtility.data.cpu().numpy()),"MSE at T (on S)":FBSDEloss_trainbyUtility.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"FBSDE","E(Utility)":-Utilityloss_trainbyFBSDE.data.cpu().numpy(),"sd(Utility)":"{:e}".format(UtilitySD_trainbyFBSDE.data.cpu().numpy()),"MSE at T (on S)":FBSDEloss_trainbyFBSDE.data.cpu().numpy()},ignore_index=True)
df=df.append({"Method":"Leading Order","E(Utility)":-UtilityLoss_APP_Mathematica.data.cpu().numpy(),"sd(Utility)":"{:e}".format(UtilitySD_APP_Mathematica.data.cpu().numpy()),"MSE at T (on S)":FBSDELoss_APP_Mathematica.data.cpu().numpy()},ignore_index=True)
df.to_csv(path+"trading{}_cost{}.csv".format(TIME,q), index=False, header=True)
