# Deep Learning Algorithms for Hedging with Frictions

This repository contains the Forward-Backward Stochastic Differential Equation (FBSDE) solver, the Deep Hedging, and the ST-Hedging, as described in reference [2]. All of them are implemented in PyTorch.

The special case with following assumptions is considered:

* the dynamic of the market satisfies that return <img src="https://latex.codecogs.com/svg.latex?\mu" /> and voalatility <img src="https://latex.codecogs.com/svg.latex?\sigma" /> are constant;
* the cost parameter <img src="https://latex.codecogs.com/svg.latex?\lambda" /> is constant;
* the endowment volatility is in the form of <img src="https://latex.codecogs.com/svg.latex?\xi_t={\xi}W_t" /> where <img src="https://latex.codecogs.com/svg.latex?{\xi}" /> is constant; 
* the frictionless strategy satisfies that   <img src="https://latex.codecogs.com/svg.latex?\bar{b}_t=0" /> and <img src="https://latex.codecogs.com/svg.latex?\bar{a}_t=-{\xi}/{\sigma}" />

## Basic Setup for the case with a single stock

We consider two calibrated models: a quadratic transaction cost models, and a power cost model with elastic parameter of 3/2. In both experiments, the FBSDE solver, the Deep Hedging, and the ST-Hedging, are implemented, as well as the asymptotic formula from Equation (3.4) in reference [2].     
<br/>
For the case of quadratic costs, the ground truth from equation (4.1) in reference [2] is also compared. See [`Script/sample_code_quadratic_cost.py`](./Script/sample_code_quadratic_cost.py) for details.   
<br/>
For the case of 3/2 power costs, the ground truth is no longer available in closed form. Meanwhile, in regard to the asymptotic formula g(x) in equation (3.5) in reference [2], the numerical solution by [SciPy](https://github.com/scipy/scipy) is not stable, thus it is solved via MATHEMATICA (see [`Script/power_cost_ODE.nb`](./Script/power_cost_ODE.nb)). Consequently, the value of g(x) corresponding to x ranging from 0 to 50 by 0.0001, is stored in table [`Data/EVA.txt`](./Data/EVA.txt). Benefitted from the oddness and the growth conditions (equation (A.5) in reference [2]), the value of g(x) on <img src="https://latex.codecogs.com/svg.latex?\mathbb{R}" /> is obatinable. Following that, the numerical result of the asymptotic solution is compared with two machine learning methods. See [`Script/sample_code_power_cost.py`](./Script/sample_code_power_cost.py) for details.
<br/><br/>
The general variables and the market parameters in the code are summarized below:
| Variable | Meaning |
| --- | --- |
| `q`  | power of the trading cost, q |
| `S_OUTSTANDING` | total shares in the market, s |
| `TIME` | trading horizon, T |
| `TIME_STEP` |   time discretization, N |
| `DT ` | <img src="https://latex.codecogs.com/svg.latex?\Delta%20t={T}/{N}" />  |
| `GAMMA` | risk aversion, <img src="https://latex.codecogs.com/svg.latex?\gamma" /> |
| `XI_1` | endowment volatility parameter, <img src="https://latex.codecogs.com/svg.latex?{\xi}" /> |
| `PHI_INITIAL` | initial holding,  <img src="https://latex.codecogs.com/svg.latex?\varphi_{0-}" /> |
| `ALPHA` | market volatility,  <img src="https://latex.codecogs.com/svg.latex?\sigma " /> |
| `MU_BAR` | market return,  <img src="https://latex.codecogs.com/svg.latex?\mu " /> |
| `LAM` | trading cost parameter, <img src="https://latex.codecogs.com/svg.latex?\lambda " /> |
| `test_samples` | number of test sample path, batch_size |

## Basic Setup for the case with multiple stocks

For high dimensional case with three stocks, we consider the quadratic transaction cost model. The asymptotic formula from Equation (3.4) in reference [2], and the ground truth from equation (4.1) in reference [2] are included in `leading_order_quad` and `ground_truth` of `DynamicsFactory` class in 
[`SingleAgent-Stage2/SingleAgentPipe.py`](./SingleAgent-Stage2/SingleAgentPipe.py). And we implement the ST-Hedging algorithm illustrated in Section (4.3) in reference [2]. 
<br/><br/>
The general variables and the market parameters in the code are summarized below:
| Variable | Meaning |
| --- | --- |
| `N_STOCK` | dimension of the stocks |
| `S_OUTSTANDING` | total shares in the market, s |
| `TR` | trading horizon, T |
| `T` |   time discretization, N |
| `BM_COV` |  Covariance matrix of the high dimensional Brownian Motion,  an identity matrix. |
| `GAMMA` | risk aversion, <img src="https://latex.codecogs.com/svg.latex?\gamma" /> |
| `xi_dd` | endowment volatility parameter, <img src="https://latex.codecogs.com/svg.latex?{\xi}" /> |
| `alpha_stmd` | market volatility,  <img src="https://latex.codecogs.com/svg.latex?\sigma " /> |
| `mu_stm` | market return,  <img src="https://latex.codecogs.com/svg.latex?\mu " /> |
| `lam_mm` | trading cost parameter, <img src="https://latex.codecogs.com/svg.latex?\Lambda " /> |
| `N_SAMPLE` | number of test sample path, batch_size |


## FBSDE solver
### Case for the Single Stock
For the detailed implementation of the FBSDE solver, see [`Script/sample_code_FBSDE.py`](./Script/sample_code_FBSDE.py);      
The core dynamic is defined in the method `System.forward()`, and the key variables in the code are summarized below:   
| Variable | Meaning |
| --- | --- |
| `time_step` | time discretization, N |
| `n_samples` | number of sample path, batch_size |
| `dW_t`  | iid normally distributed random variables with mean zero and variance <img src="https://latex.codecogs.com/svg.latex?\Delta%20t" />, <img src="https://latex.codecogs.com/svg.latex?\Delta%20W_t" /> |
| `W_t` | Brownian motion at time t, <img src="https://latex.codecogs.com/svg.latex?W_t" />  |
|  `XI_t` | Brownian motion at time t, <img src="https://latex.codecogs.com/svg.latex?W_t" /> |
| `sigma_t` | vector of 0 |
|  `sigmaxi_t` | vector of 1 |
|  `X_t` | vector of 1 |
|  `Y_t` | vector of 0 |
| `Lam_t` | 1 |
|  `in_t` | input of the neural network <img src="https://latex.codecogs.com/svg.latex?F^{\theta} " /> |
|   `sigmaZ_t` | output of the neural network <img src="https://latex.codecogs.com/svg.latex?F^{\theta} " />,  <img src="https://latex.codecogs.com/svg.latex?Z_{t} " /> |
| `Delta_t` | difference between the frictional and frictionless positions (the **forward component**) divided by the endowment parameter, <img src="https://latex.codecogs.com/svg.latex?\Delta%20\varphi_t/\xi" /> |
| `Z_t` | the **backward component**, <img src="https://latex.codecogs.com/svg.latex?Y_t " /> |

### Case for Multiple Stocks
For the detailed implementation of the FBSDE solver, see the class `DynamicsFactory` in 
[`SingleAgent-Stage2/SingleAgentPipe.py`](./SingleAgent-Stage2/SingleAgentPipe.py);      
The core dynamic is defined in the function `fbsde_quad`, and the key variables in the code are summarized below:   
| Variable | Meaning |
| --- | --- |
| `dW_std`  | iid normally distributed random variables with mean zero and variance <img src="https://latex.codecogs.com/svg.latex?\Delta%20t" />, <img src="https://latex.codecogs.com/svg.latex?\Delta%20W_t" /> |
| `W_std` | 3-dimsion Brownian motion at time t, <img src="https://latex.codecogs.com/svg.latex?W_t" />  |
| `xi_std_w` | collection of the endowment volatility, throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\xi_t\}" /> |
|  `x` | input of the neural network <img src="https://latex.codecogs.com/svg.latex?F^{\theta} " /> |
|   `Z_stmd` | output of the neural network <img src="https://latex.codecogs.com/svg.latex?F^{\theta} " />,  <img src="https://latex.codecogs.com/svg.latex?Z_{t} " /> |
| `phi_stm` | collection of the frictional positions throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\varphi_t\}" /> |
| `phi_stm_bar` | collection of the frictionless positions throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\bar{\varphi_t}\}" /> |
| `phi_dot_stm` | collection of the frictional trading rate throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\dot{\varphi_t}\}" /> |


## Deep Hedging
### Case for the Single Stock
For the detailed implementation of the Deep Hedging, see [`Script/sample_code_Deep_Hedging.py`](./Script/sample_code_Deep_Hedging.py);   
The core dynamic of the Deep Hedging is defined in the function `TRAIN_Utility()`, and the key variables in the code are summarized below:
| Variable | Meaning |
| --- | ---  |
| `time_step`  |   time discretization, N |
| `n_samples` | number of sample path, batch_size |
| `PHI_0_on_s` | initial holding divided by the total shares in the market, <img src="https://latex.codecogs.com/svg.latex?\varphi_{0-}/s" /> |
| `W` | collection of the Brownian motion, throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{W_t\}" /> |
| `XI_W_on_s` | collection of the endowment volatility divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\xi_t/s\}" /> |
| `PHI_on_s` | collection of the frictional positions divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\varphi_t/s\}" /> |
| `PHI_dot_on_s` | collection of the frictional trading rate divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\dot{\varphi_t}/s\}" /> |
| `loss_Utility` | minus goal function, <img src="https://latex.codecogs.com/svg.latex?-J_T(\dot{\varphi})" /> |

### Case for Multiple Stocks
For the detailed implementation of the Deep Hedging, see the class `DynamicsFactory` in 
[`SingleAgent-Stage2/SingleAgentPipe.py`](./SingleAgent-Stage2/SingleAgentPipe.py);      
The core dynamic is defined in the function `deep_hedging`, and the key variables in the code are summarized below:   
| Variable | Meaning |
| --- | ---  |
| `dW_std`  | iid normally distributed random variables with mean zero and variance <img src="https://latex.codecogs.com/svg.latex?\Delta%20t" />, <img src="https://latex.codecogs.com/svg.latex?\Delta%20W_t" /> |
| `W_std` | 3-dimsion Brownian motion at time t, <img src="https://latex.codecogs.com/svg.latex?W_t" />  |
| `xi_std_w` | collection of the endowment volatility, throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\xi_t\}" /> |
|  `x` | input of the neural network <img src="https://latex.codecogs.com/svg.latex?F^{\theta} " /> |
| `phi_stm` | collection of the frictional positions throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\varphi_t\}" /> |
| `phi_stm_bar` | collection of the frictionless positions throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\bar{\varphi_t}\}" /> |
| `phi_dot_stm` | output of the neural network <img src="https://latex.codecogs.com/svg.latex?F^{\theta} " />, collection of the frictional trading rate throughout the trading horizon, <img src="https://latex.codecogs.com/svg.latex?\{\dot{\varphi_t}\}" /> |

## ST-Hedging Algorithm
For the detailed implementation of the ST-Hedging Algorithm, see the class `DynamicsFactory` in 
[`SingleAgent-Stage2/SingleAgentPipe.py`](./SingleAgent-Stage2/SingleAgentPipe.py);      
The core dynamic is defined in the function `st_hedging`, and the key variables in the code are summarized below:   
| Variable | Meaning |
| --- | ---  |
| `M`  | cut-off value for the trading horizon considered as long enough for the leading-order solution   |
| `phi_stm` | collection of the frictional positions throughout the trading horizon: taking the value of the leading-order solution from time 0 to M, and the value of the Deep Hedging from time M to the end of trading horizon, T |
| `phi_dot_stm` | collection of the frictional trading rate throughout the trading horizon, taking the value of the leading-order solution from time 0 to M, and the value of the Deep Hedging from time M to the end of trading horizon, T|
| `phi_stm_leading_order` | collection of the frictional positions from the initial time to time M, given by the leading-order solution|
| `phi_dot_stm_leading_order` | collection of the frictional trading rate from the initial time to time M, given by the leading-order solution|
| `phi_stm_deep_hedging` | collection of the frictional positions from time M to the end of trading horizon, given by the deep hedging|
| `phi_dot_stm_deep_hedging` | collection of the frictional trading rate from time M to the end of trading horizon, given by the deep hedging|
| `phi_0` | initial value of frictional position for the deep heding, given by the leading-order formula at the cut-off time at time M,  <img src="https://latex.codecogs.com/svg.latex?\{\varphi_M\}" /> |




## Example  
### Single Stock
#### Quatratic Cost
Here we proivde an example for the quadratic cost case (`q=2`) with the trading horizon of 21 days (`TIME=21`).    
<br/>
The trading horizon is discretized in 168 time steps (`TIME_STEP=168`). The parameters are taken from the calibration in [1]:

| Parameter | Value | Code | 
| --- | ---  | --- | 
| agent risk aversion  | <img src="https://latex.codecogs.com/svg.latex?\gamma=1.66\times10^{-13}"/> | `GAMMA=1.66*1e-13` | 
|total shares outstanding |<img src="https://latex.codecogs.com/svg.latex?s=2.46\times10^{11}"/> | `S_OUTSTANDING=2.46*1e11` |
|stock volatility  |<img src="https://latex.codecogs.com/svg.latex?\sigma=1.88"/>  | `ALPHA=1.88`|
| stock return|<img src="https://latex.codecogs.com/svg.latex?\mu=0.5\times\gamma\times\sigma^2"/> |`MU_BAR=0.5*GAMMA*ALPHA**2` |
| endowment volatility parameter | <img src="https://latex.codecogs.com/svg.latex?{\xi}=2.19\times10^{10}" />| `XI_1=2.19*1e10` |
| trading cost parameter |<img src="https://latex.codecogs.com/svg.latex?\lambda=1.08\times10^{-10}"/> | `LAM=1.08*1e-10`|

And these lead to the optimal trading rate (left panel) and the optimal position (right panel) illustrated below, leanrt by the FBSDE solver, the Deep Hedging, and the ST-Hedging as well as the ground truth and the Leading-order solution based on the asymptotic formula:   

<!--[TR=21_q=2](./Gallery/single_quad_21.png)-->
![TR=21_q=2](./Gallery/single_quad_21_new.png)
<br/>
With the simulation of a test batch size of 3000 (`test_samples=3000`), the expectation and the standard deviation of the goal function <img src="https://latex.codecogs.com/svg.latex?J_T(\dot{\varphi})"/> and the mean square error of the terminal trading rate are calculated, as summarized below:

| Method | <img src="https://latex.codecogs.com/svg.image?J_T(\dot{\varphi})\pm%20\mathrm{std}"/> | <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[(\dot{\varphi_T})^2/s^2]"/> | 
| --- | ---  | --- | 
| FBSDE Solver  | <img src="https://latex.codecogs.com/svg.image?4.13\times&space;10^9\pm&space;2.20&space;\times&space;10^9"/> | <img src="https://latex.codecogs.com/svg.image?1.35\times&space;10^{-8}"/> | 
| Deep Hedging  | <img src="https://latex.codecogs.com/svg.image?4.13\times&space;10^9\pm&space;2.20&space;\times&space;10^9"/> | <img src="https://latex.codecogs.com/svg.image?3.62\times&space;10^{-9}"/> | 
| ST Hedging  | <img src="https://latex.codecogs.com/svg.image?4.13\times&space;10^9\pm&space;2.19&space;\times&space;10^9"/> | <img src="https://latex.codecogs.com/svg.image?1.26&space;\times&space;10^{-8}"/> | 
| Leading Order Approximation  |  <img src="https://latex.codecogs.com/svg.image?4.06\times&space;10^9\pm&space;2.21&space;\times&space;10^9"/> | <img src="https://latex.codecogs.com/svg.image?7.89&space;\times&space;10^{-5}"/>| 
| Ground Truth |  <img src="https://latex.codecogs.com/svg.image?4.13\times&space;10^9\pm&space;2.20&space;\times&space;10^9"/> | <img src="https://latex.codecogs.com/svg.image?0.0"/>| 

#### 3/2 Power Cost
Here we proivde an example for the 3/2 power cost case (`q=3/2`) with the trading horizon of 21 days (`TIME=21`).    
<br/>
The trading horizon is discretized in 168 time steps (`TIME_STEP=168`). The parameters are taken from the calibration in [1]:

| Parameter | Value | Code | 
| --- | ---  | --- | 
| agent risk aversion  | <img src="https://latex.codecogs.com/svg.latex?\gamma=1.66\times10^{-13}"/> | `GAMMA=1.66*1e-13` | 
|total shares outstanding |<img src="https://latex.codecogs.com/svg.latex?s=2.46\times10^{11}"/> | `S_OUTSTANDING=2.46*1e11` |
|stock volatility  |<img src="https://latex.codecogs.com/svg.latex?\sigma=1.88"/>  | `ALPHA=1.88`|
| stock return|<img src="https://latex.codecogs.com/svg.latex?\mu=0.5\times\gamma\times\sigma^2"/> |`MU_BAR=0.5*GAMMA*ALPHA**2` |
| endowment volatility parameter | <img src="https://latex.codecogs.com/svg.latex?{\xi}=2.19\times10^{10}" />| `XI_1=2.19*1e10` |
| trading cost parameter |<img src="https://latex.codecogs.com/svg.image?5.22&space;\times&space;10^{-6}"/> | `LAM=5.22e-6`|

And these lead to the optimal trading rate (left panel) and the optimal position (right panel) illustrated below, leanrt by the FBSDE solver, the Deep Hedging, and the ST-Hedging as well as the Leading-order solution based on the asymptotic formula:   

![TR=21_q=3/2](./Gallery/single_3_on_2_21_new.png)
<br/>
With the simulation of a test batch size of 3000 (`test_samples=3000`), the expectation and the standard deviation of the goal function <img src="https://latex.codecogs.com/svg.latex?J_T(\dot{\varphi})"/> and the mean square error of the terminal trading rate are calculated, as summarized below:

| Method | <img src="https://latex.codecogs.com/svg.image?J_T(\dot{\varphi})\pm%20\mathrm{std}"/> | <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[(\dot{\varphi_T})^2/s^2]"/> | 
| --- | ---  | --- | 
| FBSDE Solver  | <img src="https://latex.codecogs.com/svg.image?4.02&space;\times&space;10^{9}&space;\pm&space;2.42&space;\times&space;10^{9}&space;"/> | <img src="https://latex.codecogs.com/svg.image?4.55&space;\times&space;10^{-9}"/> | 
| Deep Hedging  | <img src="https://latex.codecogs.com/svg.image?4.02&space;\times&space;10^{9}&space;\pm&space;2.42&space;\times&space;10^{9}&space;"/> | <img src="https://latex.codecogs.com/svg.image?1.68&space;\times&space;10^{-9}"/> | 
| ST Hedging  | <img src="https://latex.codecogs.com/svg.image?4.02&space;\times&space;10^{9}&space;\pm&space;2.40&space;\times&space;10^{9}&space;"/> | <img src="https://latex.codecogs.com/svg.image?1.34&space;\times&space;10^{-10}"/> | 
| Leading Order Approximation  |  <img src="https://latex.codecogs.com/svg.image?3.93&space;\times&space;10^{9}&space;\pm&space;2.42&space;\times&space;10^{9}&space;"/> | <img src="https://latex.codecogs.com/svg.image?1.10&space;\times&space;10^{-4}"/>| 

### Multiple Stocks
To illustrate the scalability of our ST-Hedging algorithm, we proivde an example with three risky assets in the market with cross sectional effect, under the quadratic cost case (`q=2`).
The trading horizon is 2520 days (`TR=2520`), discretized in 2520 time steps (`T=2520`), and the switching threshold is 100 days before maturity.

 The parameters are taken from the calibration in [1]:

| Parameter | Value | Code | 
| --- | ---  | --- | 
| agent risk aversion  | <img src="https://latex.codecogs.com/svg.image?7.424&space;\times&space;10^{-13}"/> | `GAMMA = 1/(1/ (8.91*1e-13) + 1/ (4.45 * 1e-12) )` | 
|total shares outstanding |<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;1.15&space;\times&space;10^{10}\\&space;3.2&space;\times&space;10^9&space;\\&space;2.3&space;\times&space;10^9\end{bmatrix}"/> | `S_OUTSTANDING = torch.tensor([1.15, 0.32, 0.23]) *1e10` |
|stock volatility  |<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;72.00&&space;71.49&space;&&space;54.80&space;\\&space;71.49&space;&&space;85.42&space;&&space;65.86&space;\\&space;54.80&&space;65.86&space;&&space;&space;56.84\\\end{bmatrix}"/>  | `sigma_big = torch.tensor([[72.00, 71.49, 54.80],[71.49, 85.42, 65.86],[54.80, 65.86, 56.84]])`|
| stock return|<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;2.99\\&space;3.71&space;\\&space;3.55\end{bmatrix}"/> |`mu_stm = torch.ones((n_sample, time_len, N_STOCK)) * torch.tensor([[2.99, 3.71, 3.55]])` |
| endowment volatility parameter | <img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;-2.07&&space;1.91&space;&&space;0.64&space;\\&space;1.91&&space;&space;-1.77&&space;-0.59&space;\\&space;0.64&&space;-0.59&space;&space;&&space;&space;-0.20\\\end{bmatrix}\times&space;10^9" />| `xi_dd = torch.tensor([[ -2.07, 1.91, 0.64],[1.91, -1.77, -0.59],[0.64 ,-0.59 ,-0.20]]) *1e9` |
| trading cost parameter |<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;1.269&space;\\&space;1.354&space;\\&space;1.595\end{bmatrix}&space;\times&space;10^{-9}"/> | ` lam_mm = torch.diag(torch.tensor([0.1269, 0.1354, 0.1595])) * 1e-8`|

And these lead to the optimal position (the first plot) and the optimal trading rates illustrated below (we only include the tradings in the last 30 days), leanrt by the ST-Hedging as well as the ground truth and the Leading-order solution based on the asymptotic formula:   
![multiple stocks position](./Gallery/quad_phi_10yr.png "position") ![multiple stocks rate1](./Gallery/quad_phidot_10yr_1.png "rate 1")
![multiple stocks rate2](./Gallery/quad_phidot_10yr_2.png "rate 2") ![multiple stocks rate3](./Gallery/quad_phidot_10yr_3.png "rate 3")
<br/>
With the simulation of a test batch size of 3000 (`N_SAMPLE = 3000`), the expectation and the standard deviation of the goal function <img src="https://latex.codecogs.com/svg.latex?J_T(\dot{\varphi})"/> and the mean square error of the terminal trading rate are calculated, as summarized below:

| Method | <img src="https://latex.codecogs.com/svg.image?J_T(\dot{\varphi})\pm%20\mathrm{std}"/> | <img src="https://latex.codecogs.com/svg.image?\mathbb{E}[(\dot{\varphi_T})^2/s^2]"/> | 
| --- | ---  | --- | 
| ST Hedging  | <img src="https://latex.codecogs.com/svg.image?1.60&space;\times&space;10^{11}&space;\pm&space;3.58&space;\times&space;10^6"/> | <img src="https://latex.codecogs.com/svg.image?\begin{pmatrix}&space;2.4&&space;4.7&space;&3.6&space;&space;\\\end{pmatrix}&space;\times&space;10^{-9}"/> | 
| Leading Order Approximation  |  <img src="https://latex.codecogs.com/svg.image?1.60&space;\times&space;10^{11}&space;\pm&space;3.58&space;\times&space;10^6"/> | <img src="https://latex.codecogs.com/svg.image?\begin{pmatrix}&space;4.5&&space;3.5&space;&1.6&space;&space;\\\end{pmatrix}&space;\times&space;10^{-7}"/>| 
| Ground Truth |  <img src="https://latex.codecogs.com/svg.image?1.60&space;\times&space;10^{11}&space;\pm&space;3.58&space;\times&space;10^6"/> | <img src="https://latex.codecogs.com/svg.image?\begin{pmatrix}&space;0.0&&space;0.0&space;&0.0&space;&space;\\\end{pmatrix}&space;"/>| 

See more examples and discussion in Section 4 of paper [2].   

<!-- 
## Authors

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
 -->
## Acknowledgments

## Reference
[1]  Asset Pricing with General Transaction Costs: Theory and Numerics, L. Gonon, J. Muhle-Karbe, X. Shi. [[Mathematical Finance]](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.12297), 2021.

[2]  Deep Learning Algorithms for Hedging with Frictions, X. Shi, D. Xu, Z. Zhang. [[arXiv]](https://arxiv.org/abs/2111.01931#), 2021. 
