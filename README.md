# Deep Learning Algorithms for Hedging with Frictions

This repository contains the Forward-Backward Stochastic Differential Equation (FBSDE) solver and the Deep Hedging, as described in reference [2]. Both of them are implemented in PyTorch.

## Basic Setup

The special case with following assumptions is considered:

* the dynamic of the market satisfies that return <img src="https://latex.codecogs.com/gif.latex?\mu" /> and voalatility <img src="https://latex.codecogs.com/gif.latex?\sigma" /> are constant;
* the cost parameter <img src="https://latex.codecogs.com/gif.latex?\lambda" /> is constant;
* the endowment volatility is in the form of <img src="https://latex.codecogs.com/gif.latex?\xi_t=\hat{\xi}W_t" /> where <img src="https://latex.codecogs.com/gif.latex?\hat{\xi}" /> is constant; 
* the frictionless strategy satisfies that   <img src="https://latex.codecogs.com/gif.latex?\bar{b_t}=0" /> and <img src="https://latex.codecogs.com/gif.latex?\bar{a_t}=-\hat{\xi}{\sigma}^{-1}" />

On top of that, we consider two calibrated models: a quadratic transaction cost models, and a power cost model with elastic parameter of 3/2. In both experiments, the FBSDE solver and the Deep Hedging are implemented, as well as the asymptotic formula from Theorem 3.6 in reference [2].     
<br/>
For the case of quadratic costs, the ground truth from equation (3.7) in reference [2] is also compared. See [`Script/sample_code_quadratic_cost.py`](./Script/sample_code_quadratic_cost.py) for details.   
<br/>
For the case of 3/2 power costs, the ground truth is no longer available in closed form. Meanwhile, in regard to the asymptotic formula g(x) in equation (3.8) in reference [2], the numerical solution by [SciPy](https://github.com/scipy/scipy) is not stable, thus it is solved via MATHEMATICA (see [`Script/3on2cost_ODE.nb`](./Script/3on2cost_ODE.nb)). Consequently, the value of g(x) corresponding to x ranging from 0 to 50 by 0.0001, is stored in table [`Data/EVA.txt`](./Data/EVA.txt). Benefitted from the oddness and the growth conditions (equation (3.9) in reference [2]), the value of g(x) on <img src="https://latex.codecogs.com/gif.latex?\mathbb{R}" /> is obatinable. Following that, the numerical result of the asymptotic solution is compared with two machine learning methods. See [`Script/sample_code_3_2_cost.py`](./Script/sample_code_3_2_cost.py) for details.
<br/><br/>
The general variables and the market parameters in the code are summarized below:
| Variable | Meaning |
| --- | --- |
| `q`  | power of the trading cost, q |
| `S_OUTSTANDING` | total shares in the market, s |
| `TIME` | trading horizon, T |
| `TIME_STEP` |   time discretization, N |
| `DT ` | <img src="https://latex.codecogs.com/gif.latex?\Delta%20t=\frac{T}{N}" />  |
| `GAMMA` | risk aversion, <img src="https://latex.codecogs.com/gif.latex?\gamma" /> |
| `XI_1` | endowment volatility parameter, <img src="https://latex.codecogs.com/gif.latex?\hat{\xi}" /> |
| `PHI_INITIAL` | initial holding,  <img src="https://latex.codecogs.com/gif.latex?\varphi_{0-}" /> |
| `ALPHA` | market volatility,  <img src="https://latex.codecogs.com/gif.latex?\sigma " /> |
| `MU_BAR` | market return,  <img src="https://latex.codecogs.com/gif.latex?\mu " /> |
| `LAM` | trading cost parameter, <img src="https://latex.codecogs.com/gif.latex?\lambda " /> |
| `test_samples` | number of test sample path, batch_size |

## FBSDE solver
For the detailed implementation of the FBSDE solver, see [`Script/sample_code_FBSDE.py`](./Script/sample_code_FBSDE.py);      
The core dynamic is defined in the method `System.forward()`, and the key variables in the code are summarized below:   
| Variable | Meaning |
| --- | --- |
| `time_step` | time discretization, N |
| `n_samples` | number of sample path, batch_size |
| `dW_t`  | iid normally distributed random variables with mean zero and variance <img src="https://latex.codecogs.com/gif.latex?\Delta%20t" />, <img src="https://latex.codecogs.com/gif.latex?\Delta%20W_t" /> |
| `W_t` | Brownian motion at time t, <img src="https://latex.codecogs.com/gif.latex?W_t" />  |
|  `XI_t` | Brownian motion at time t, <img src="https://latex.codecogs.com/gif.latex?W_t" /> |
| `sigma_t` | vector of 0 |
|  `sigmaxi_t` | vector of 1 |
|  `X_t` | vector of 1 |
|  `Y_t` | vector of 0 |
| `Lam_t` | 1 |
|  `in_t` | input of the neural network <img src="https://latex.codecogs.com/gif.latex?F^{\theta} " /> |
|   `sigmaZ_t` | output of the neural network <img src="https://latex.codecogs.com/gif.latex?F^{\theta} " />,  <img src="https://latex.codecogs.com/gif.latex?Z_{t} " /> |
| `Delta_t` | difference between the frictional and frictionless positions (the **forward component**) divided by the endowment parameter, <img src="https://latex.codecogs.com/gif.latex?{\hat{\xi}}^{-1}\Delta%20\varphi_t " /> |
| `Z_t` | the **backward component**, <img src="https://latex.codecogs.com/gif.latex?Y_t " /> |


## Deep Hedging
For the detailed implementation of the Deep Hedging, see [`Script/sample_code_Deep_Hedging.py`](./Script/sample_code_Deep_Hedging.py);   
The core dynamic of the Deep Hedging is defined in the function `TRAIN_Utility()`, and the key variables in the code are summarized below:
| Variable | Meaning |
| --- | ---  |
| `time_step`  |   time discretization, N |
| `n_samples` | number of sample path, batch_size |
| `PHI_0_on_s` | initial holding divided by the total shares in the market, <img src="https://latex.codecogs.com/gif.latex?s^{-1}\times\varphi_{0-}" /> |
| `W` | collection of the Brownian motion, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{W_t\}" /> |
| `XI_W_on_s` | collection of the endowment volatility divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{s^{-1}\times\xi_t\}" /> |
| `PHI_on_s` | collection of the frictional positions divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{s^{-1}\times\varphi_t\}" /> |
| `PHI_dot_on_s` | collection of the frictional trading rate divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{s^{-1}\times\dot{\varphi_t}\}" /> |
| `loss_Utility` | minus goal function, <img src="https://latex.codecogs.com/gif.latex?-J_T(\dot{\varphi})" /> |

## Example
Here we proivde an example for the quadratic cost case (`q=2`) with the trading horizon of 21 days (`TIME=21`).    
<br/>
The trading horizon is discretized in 168 time steps (`TIME_STEP=168`). The parameters are taken from the calibration in [1]:

| Parameter | Value | Code | 
| --- | ---  | --- | 
| agent risk aversion  | <img src="https://latex.codecogs.com/gif.latex?\gamma=1.66\times10^{-13}"/> | `GAMMA=1.66*1e-13` | 
|total shares outstanding |<img src="https://latex.codecogs.com/gif.latex?s=2.46\times10^{11}"/> | `S_OUTSTANDING=2.46*1e11` |
|stock volatility  |<img src="https://latex.codecogs.com/gif.latex?\sigma=1.88"/>  | `ALPHA=1.88`|
| stock return|<img src="https://latex.codecogs.com/gif.latex?\mu=0.5\times\gamma\times\sigma^2"/> |`MU_BAR=0.5*GAMMA*ALPHA**2` |
| endowment volatility parameter | <img src="https://latex.codecogs.com/gif.latex?\hat{\xi}=2.19\times10^{10}" />| `XI_1=2.19*1e10` |
| trading cost parameter |<img src="https://latex.codecogs.com/gif.latex?\lambda=1.08\times10^{-10}"/> | `LAM=1.08*1e-10`|

And these lead to the optimal trading rate (left panel) and the optimal position (right panel) illustrated below, leanrt by the FBSDE solver and the Deep Hedging, as well as the ground truth and the Leading-order solution based on the asymptotic formula:   

![TR=21_q=2](./Gallery/single_quad_21.png)
<br/>
With the same simulation with test batch size of 3000 (`test_samples=3000`), the expectation and the standard deviation of the goal function <img src="https://latex.codecogs.com/gif.latex?J_T(\dot{\varphi})"/> and the mean square error of the terminal trading rate are calculated, as summarized below:

| Method | <img src="https://latex.codecogs.com/gif.latex?J_T(\dot{\varphi})\pm%20\mathrm{std}"/> | <img src="https://latex.codecogs.com/gif.latex?\mathbb{E}[(\dot{\varphi_T})^2/s^2]"/> | 
| --- | ---  | --- | 
| FBSDE  | <img src="https://latex.codecogs.com/gif.latex?4.11\times10^9\pm%202.20\times10^9"/> | <img src="https://latex.codecogs.com/gif.latex?1.61\times10^{-8}"/> | 
| Deep Q-learning  | <img src="https://latex.codecogs.com/gif.latex?4.13\times10^9\pm%202.20\times10^9"/> | <img src="https://latex.codecogs.com/gif.latex?3.62\times10^{-9}"/> | 
| Leading Order Approximation  |  <img src="https://latex.codecogs.com/gif.latex?4.06\times10^9\pm%202.21\times10^9"/> | <img src="https://latex.codecogs.com/gif.latex?7.89\times10^{-5}"/>| 
| Ground Truth |  <img src="https://latex.codecogs.com/gif.latex?4.13\times10^9\pm%202.20\times10^9"/> | <img src="https://latex.codecogs.com/gif.latex?1.25\times10^{-8}"/>| 



See more examples and discussion in Section 4 of paper [2].   

<!-- 
## Authors

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
 -->
## Acknowledgments

## Reference
[1]  Asset Pricing with General Transaction Costs: Theory and Numerics, L. Gonon, J. Muhle-Karbe, X. Shi. [[Mathematical Finance]](https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.12297), 2021.

[2]  Deep Learning Algorithms for Hedging with Frictions, X. Shi, D. Xu, Z. Zhang. [[arXiv]](https://arxiv.org), 2021. 
