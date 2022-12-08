# SingleAgentPipe.py Usage
This script solves the optimal hedging strategies of a single agent under high-dimensional settings. It can take into arbitrary number of stocks, with arbitary Brownian motion dimensions. It gives researchers the flexibility to train the deep hedging, FBSDE, ST-Hedging algorithms using deep neural networks, or simulate Brownian paths using leading order approximations and ground truth dynamics by simply tweaking a few parameters at the end of the file. The parameters and their usages are illustrated in the table below.

| Parameter | Description | Options |
| --- | --- | --- |
| algo| The algorithm to train using deep neural networks | <strong>deep_hedging</strong>: The deep hedging algorithm that parameterizes the entire optimal hedging policy using deep neural networks.<br><strong>fbsde</strong>: The FBSDE framework where the diffusion of the transaction rate is parameterized by deep neural networks.<br><strong>pasting</strong>: The algorithm that pastes the leading order approximation in the first half and deep_hedging in the second half. |
| cost| The cost function for transactions | <strong>quadratic</strong>: The transaction cost is in the power of 2.<br><strong>power</strong>: The transaction cost is in the power of 3/2. |
| model_name | The architecture for the deep neural networks | <strong>discretized_feedforward</strong>: A list of shallow neural networks discretized at each timestamp.<br><strong>rnn</strong>: A holistic recurrent neural network that is not discretized by time. |
| solver | The solver to use for optimization | <strong>Adam</strong>: The Adam solver.<br><strong>RMSprop</strong>: The RMSprop solver.<br><strong>SGD</strong>: The stochastic gradient descent solver. |
| hidden_lst | The hidden layers for each shallow neural network | A list of positive integers. E.g. [50, 50, 50] |
| lr | The learning rate of the algorithm | A positive real number. E.g. 1e-2 |
| epoch | The number of training epochs | A positive integer. E.g. 10000 |
| decay | The decay rate of the learning rate at each step of the scheduler | A positive real number. E.g. 0.1 |
| scheduler_step | The number of epochs for each step in the scheduler | A positive integer. E.g. 5000 |
| retrain | Whether to retrain a new model or reuse the latest pre-trained one | <strong>True</strong>: Ignore the old models and train a new model.<br><strong>False</strong>: Continue training on the latest pre-trained model of the same algorithm with the same model architecture. |
| pasting_cutoff | The cutoff position on the discretized timestamps for the pasting algorithm | A positive integer between 0 and T. |

<!-- 
## Authors
Zhanhao Zhang, Xiaofei Shi, Daran Xu

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details
 -->
