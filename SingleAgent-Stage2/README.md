# SingleAgentPipe.py Usage
This script solves the optimal hedging strategies of a single agent under high-dimensional settings. It can take into arbitrary number of stocks, with arbitary Brownian motion dimensions. It gives researchers the flexibility to train the deep hedging, FBSDE, pasting algorithms using deep neural networks, or simulate Browniam paths using leading order approximations and ground truth dynamics by simply tweaking a few parameters at the end of the file. The parameters and their usages are illustrated in the table below.

| Parameter | Description | Options |
| --- | --- | --- |
| algo| The algorithm to train using deep neural networks | - |
| cost| The cost function for transactions | - |
| model_name | The architecture for the deep neural networks | - |
| solver | The solver to use for optimization | - |
| hidden_lst | The hidden layers for each shallow neural network | A list of positive integers. E.g. [50, 50, 50] |
| lr | The learning rate of the algorithm | A positive real number. E.g. 1e-2 |
| epoch | The number of training epochs | A positive integer. E.g. 10000 |
| decay | The decay rate of the learning rate at each step of the scheduler | A positive real number. E.g. 0.1 |
| scheduler_step | The number of epochs for each step in the scheduler | A positive integer. E.g. 5000 |
| retrain | Whether to retrain a new model or reuse the latest pre-trained one | True: Ignore the old models and train a new model.<br>False: Continue training on the latest pre-trained model of the same algorithm with the same model architecture. |
| pasting_cutoff | The cutoff position on the discretized timestamps for the pasting algorithm | A positive integer between 0 and T. |

<!-- 
## Authors
Zhanhao Zhang, Xiaofei Shi, Daran Xu

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
 -->
