# A3C - Asynchronous Methods for Deep Reinforcement Learning

## Requirements
* keras
* tensorflow
* scikit-image
* gym

## How to run
```
python a3c_atari.py
```

## Custom environments
Look at a3c/env/atari_env.py or a3c/env/cartpole_env.py to learn how to create your own environment.

## Custom network models
Look at a3c/model/model.py to learn how to create your own networks. 

## How to run with custom environments and custom network models
Look at a3c_atari.py or a3c_cartpole.py to learn how to write your own script to train the A3C agent to play your environment.

## References
* [Mnih et al., 2016, Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [coreylynch/async-rl](https://github.com/coreylynch/async-rl)