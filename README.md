# Hedging with Transaction Costs: Machine Learning or not?

This repository contains the Forward-Backward Stochastic Differential Equation (FBSDE) solver and the Deep Q-learning, as described in reference [1]. Both of them are implemented in PyTorch.

## Basic Setup

The special case with following assumptions is considered:

* the dynamic of the market satisfies that return <img src="https://latex.codecogs.com/gif.latex?\mu" /> and voalatility <img src="https://latex.codecogs.com/gif.latex?\sigma" /> are constant;
* the cost parameter <img src="https://latex.codecogs.com/gif.latex?\lambda" /> is constant;
* the endowment volatility is in the form of <img src="https://latex.codecogs.com/gif.latex?\xi_t=\hat{\xi}*W_t" /> where <img src="https://latex.codecogs.com/gif.latex?\hat{\xi}" /> is constant; 
* the frictionless strategy satisfies that   <img src="https://latex.codecogs.com/gif.latex?\bar{b_t}=0" /> and <img src="https://latex.codecogs.com/gif.latex?\bar{a_t}=-\hat{\xi}{\sigma}^{-1}" />

On top of that, we consider two calibrated models: a quadratic transaction cost models, and a power cost model with elastic parameter of 3/2. In both experiments, the FBSDE solver and the Deep Q-learning are implemented, as well as the asymptotic formula from Theorem 3.6 in reference [1].    
For the case of quadratic costs, the ground truth from equation (3.7) in reference [1] is also compared. See [`Script/sample_code_quadratic_cost.py`](./Script/sample_code_quadratic_cost.py) for details.   
For the case of 3/2 power costs, the ground truth is no longer available in closed form. Meanwhile, the asymptotic formula is beyond the numerical accuracy of python, thus it is solved via MATHEMATICA (see [`Script/3on2cost_ODE.nb`](./Script/3on2cost_ODE.nb)). The value of solution g(x) of equation (3.8) in reference [1], corresponding to x ranging from 0 to 50 by 0.0001, is stored in table [`Data/EVA.txt`](./Data/EVA.txt). Benefitted from the oddness and the linearity when x is large (see equation (3.9) in reference [1]), the numerical result of the asymptotic solution is compared with two machine learning methods. See [`Script/sample_code_3_2_cost.py`](./Script/sample_code_3_2_cost.py) for details.

The general variables and the market parameters in the code are summarized below:
| Variable | Meaning |
| --- | --- |
| `q`  | power of the trading cost, q |
| `S_OUTSTANDING` | total shares in the market, s |
| `TIME` | trading horizon, T |
| `TIME_STEP` |   time discretization, N |
| `DT ` | <img src="https://latex.codecogs.com/gif.latex?\Delta%20t=\frac{T}{N}" />  |
| `GAMMA` | risk aversion, <img src="https://latex.codecogs.com/gif.latex?\gamma" /> |
| `XI_1` | endowment parameter, <img src="https://latex.codecogs.com/gif.latex?\hat{\xi}" /> |
| `PHI_INITIAL` | initial holding,  <img src="https://latex.codecogs.com/gif.latex?\varphi_{0-}" /> |
| `ALPHA` | market volatility,  <img src="https://latex.codecogs.com/gif.latex?\sigma " /> |
| `MU_BAR` | market return,  <img src="https://latex.codecogs.com/gif.latex?\mu " /> |
| `LAM` | trading cost parameter, <img src="https://latex.codecogs.com/gif.latex?\lambda " /> |


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
| `Delta_t` | difference between the frictional and frictionless positions (the **forward component**) divided by the endowment parameter, <img src="https://latex.codecogs.com/gif.latex?{\hat{\xi}}^{-1}*\Delta%20\varphi_t " /> |
| `Z_t` | the **backward component**, <img src="https://latex.codecogs.com/gif.latex?Y_t " /> |


## Deep Q-learning
For the detailed implementation of the deep Q-learning, see [`Script/sample_code_Deep_Q.py`](./Script/sample_code_Deep_Q.py);   
The core dynamic of the Deep Q-learning is defined in the function `TRAIN_Utility()`, and the key variables in the code are summarized below:
| Variable | Meaning |
| --- | ---  |
| `T`  |   time discretization, N |
| `N` | number of sample path, batch_size |
| `PHI_0_on_s` | initial holding divided by the total shares in the market, <img src="https://latex.codecogs.com/gif.latex?s^{-1}*\varphi_{0-}" /> |
| `W` | collection of the Brownian motion, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{W_t\}" /> |
| `XI_W_on_s` | collection of the endowment volatility divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{s^{-1}*\xi_t\}" /> |
| `PHI_on_s` | collection of the frictional positions divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{s^{-1}*\varphi_t\}" /> |
| `PHI_dot_on_s` | collection of the frictional trading rate divided by the total shares in the market, throughout the trading horizon, <img src="https://latex.codecogs.com/gif.latex?\{s^{-1}*\dot{\varphi_t}\}" /> |
| `loss_Utility` | minus goal function, <img src="https://latex.codecogs.com/gif.latex?-J_T(\dot{\varphi})" /> |

## Authors

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

## Reference
[1] J. Muhle-Karbe, X. Shi, D. Xu and Z. Zhang. Hedging with Transaction Costs: Machine Learning or not? *arXiv preprint* [[arXiv]](https://arxiv.org), 2021. 