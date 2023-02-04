# ODED
This repository implements the demonstration extrapolation method from <a href="https://dl.acm.org/doi/10.1145/3511808.3557357" target="_blank">Imitation Learning to Outperform Demonstrators by Directly Extrapolating Demonstrations</a>. The implementation of <a href="https://github.com/gdebie/stochastic-deep-networks" target="_blank">SDN</a> are based on their official codebase, the implementations of PPO and GAIL are based on 
<a href="https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/tree/master/a2c_ppo_acktr/algo" target="_blank">
pytorch-a2c-ppo-acktr-gail</a> by ikostrikov, and the implementation of GAIFO is based on 
<a href="https://github.com/illidanlab/opolo-code" target="_blank">opolo-code</a> by illidanlab. Please refer to `run.sh` for running the demonstration extrapolation method.

# Dependencies
1. python: 3.7.5
2. Torch: 1.3.1
3. gym: 0.15.4
