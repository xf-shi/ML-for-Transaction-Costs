# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def Mean_Utility_on_s(XI_W_on_s,PHI_on_s,PHI_dot_on_s,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME):
    """Calculate the mean of the Minus Utility
    XI_on_s: (SAMPLE_SIZE,1,TIME_STEP) 
    PHI_on_s: (SAMPLE_SIZE,TIME_STEP+1) 
    PHI_dot_on_s: (SAMPLE_SIZE,TIME_STEP) """
    C_1=S_OUTSTANDING*GAMMA/2
    C_2=(S_OUTSTANDING)*LAM/2
    totalT = XI_W_on_s.shape[-1]-1
    totalN = XI_W_on_s.shape[0]
    loss = -torch.sum(MU_BAR*PHI_on_s[:,:-1],1) + C_1*torch.sum((ALPHA*PHI_on_s[:,:-1]+XI_W_on_s[:,:-1])**2,1) + C_2*torch.sum((torch.abs(PHI_dot_on_s))**(2),1)
    loss = loss*TIME/totalT
    return torch.mean(loss)

def SD_Utility_on_s(XI_W_on_s,PHI_on_s,PHI_dot_on_s,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME):
    """Calculate the standard deviation of the Minus Utility
    XI_on_s: (SAMPLE_SIZE,1,TIME_STEP) 
    PHI_on_s: (SAMPLE_SIZE,TIME_STEP+1) 
    PHI_dot_on_s: (SAMPLE_SIZE,TIME_STEP) """
    C_1=S_OUTSTANDING*GAMMA/2
    C_2=(S_OUTSTANDING)*LAM/2
    totalT = XI_W_on_s.shape[-1]-1
    totalN = XI_W_on_s.shape[0]
    loss = -torch.sum(MU_BAR*PHI_on_s[:,:-1],1) + C_1*torch.sum((ALPHA*PHI_on_s[:,:-1]+XI_W_on_s[:,:-1])**2,1) + C_2*torch.sum((torch.abs(PHI_dot_on_s))**(2),1)
    loss = loss*TIME/totalT
    return torch.std(loss)

criterion = nn.MSELoss() # MSE Loss

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

### Networks
class Phi_Dot_0(nn.Module):
      def __init__(self):
        super(Phi_Dot_0, self).__init__()
        self.phi_Dot_0 = nn.Linear(1, 1)      
      def forward(self, x):
        return self.phi_Dot_0(x)
class RL_Net(nn.Module):
    def __init__(self,INPUT_DIM,OUTPUT_DIM,HIDDEN_DIM):
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
INPUT_DIM_Utility = 3 #The dimension of the input is 3: phi_t, XI_t, t
OUTPUT_DIM = 1

def TRAIN_Utility(train_on_gpu,path_Q,XI,PHI_INITIAL,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME,EPOCH,N,T,HIDDEN_DIM_Utility,loading,LR_Utility,saving=0,LR_Adjust=dict(),OPT_Utility="ADAM",SEED_Utility1=0,SEED_Utility2=0):
    """
    N: path size of the Brownian Motion
    T: discretization step
    loading: If True, then load the previous models from path
    LR_Utility: initial learning rate
    saving: list, to indicate at which epoch should have the models be saved to the path_temp
         saving=0 means no saving
    LR_Adjust: a dictoinary in which the value is the absolute DECAY,
          and the key is the epoch corresponding to the updated learning rate 
    OPT_Utility: "ADAM" or "SGD"
    """   
    ### loading 
    if loading:
        model_list_Utility = []
        for t in range(T):
            model = RL_Net(INPUT_DIM_Utility,OUTPUT_DIM,HIDDEN_DIM_Utility)
            model.load_state_dict(torch.load(path_Q+'Utility_para{}.pkl'.format(t),map_location=torch.device('cpu')))
            if train_on_gpu:
                model.cuda()
            model_list_Utility.append(model) 
        if OPT_Utility=="SGD":
            optimizer_Utility = torch.optim.SGD((par for model in model_list_Utility
                            for par in model.parameters()),
                                    lr=LR_Utility)
        if OPT_Utility=="ADAM":
            optimizer_Utility = torch.optim.Adam((par for model in model_list_Utility
                            for par in model.parameters()),
                            lr=LR_Utility, betas=(0.9, 0.99))
            optimizer_Utility.load_state_dict(torch.load(path_Q+"Utility_optimizer.pt",map_location=torch.device('cpu')))   
        loss_arr_Utility=torch.load(path_Q+"Utility_LOSS_arr.pt",map_location=torch.device('cpu'))    
    else:
        model_list_Utility = []
        for _ in range(T):
            model = RL_Net(INPUT_DIM_Utility,OUTPUT_DIM,HIDDEN_DIM_Utility)
            model.apply(weights_init_uniform_rule)
            if train_on_gpu:
                model.cuda()
            model_list_Utility.append(model)
        if OPT_Utility=="SGD":
            optimizer_Utility = torch.optim.SGD((par for model in model_list_Utility
                            for par in model.parameters()),
                                    lr=LR_Utility)
        if OPT_Utility=="ADAM":
            optimizer_Utility = torch.optim.Adam((par for model in model_list_Utility
                            for par in model.parameters()),
                            lr=LR_Utility, betas=(0.9, 0.99))
        loss_arr_Utility = []  
    PHI_0_on_s=torch.ones(N)*PHI_INITIAL/S_OUTSTANDING
    DUMMY_1 = torch.ones(N).reshape((N, 1))
    if train_on_gpu:
        PHI_0_on_s = PHI_0_on_s.to(device="cuda")
        DUMMY_1 = DUMMY_1.to(device="cuda")
    for epoch in tqdm(range(EPOCH)):
        ### tuning the learning rate
        if epoch in LR_Adjust.keys():
            DECAY = LR_Adjust[epoch]
            for g in optimizer_Utility.param_groups:
                g['lr'] = LR_Utility*DECAY

        ### XI_W: (SAMPLE_SIZE,TIME_STEP+1)
        W=torch.cumsum(torch.normal(0, np.sqrt(TIME*1/T), size=(N, T)), dim=1) 
        W=torch.cat((torch.zeros((N,1)),W),dim=1) 
        XI_W_on_s = XI* W /S_OUTSTANDING
        if train_on_gpu:
            XI_W_on_s = XI_W_on_s.to(device="cuda")
                    
        optimizer_Utility.zero_grad()
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
        loss_Utility = Mean_Utility_on_s(XI_W_on_s,PHI_on_s,PHI_dot_on_s,S_OUTSTANDING,GAMMA,LAM,MU_BAR,ALPHA,TIME)
        loss_arr_Utility.append(loss_Utility.data)
        loss_Utility.backward()   
        optimizer_Utility.step()     
        if np.isinf(loss_Utility.data.cpu().numpy()) or np.isnan(loss_Utility.data.cpu().numpy()):
            print("\nFAIL")
            break
        ### saving
        if saving!=0:
            if (epoch+1) in saving:   
                for i,model in enumerate(model_list_Utility):                    
                      torch.save(model.state_dict(),path_Q+'Utility_para{}.pkl'.format(i))      
                torch.save(loss_arr_Utility,path_Q+"Utility_LOSS_arr.pt")                
                torch.save(optimizer_Utility.state_dict(),path_Q+"Utility_optimizer.pt")
                print("\n saving models after {} Epochs".format(epoch+1))
    result={
        'loss':loss_arr_Utility,
        'model_list':model_list_Utility
      }
    return(result)
