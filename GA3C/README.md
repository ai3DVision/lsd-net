# GA3C: Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU

## Requirements
* python 3.4.2
* keras 2.0.2
* tensorflow 1.0.1
* gym 0.8.1

## How to run
```
python ga3c_atari.py
```

## Custom environments
Look at env/Environment.py to learn how to create your own environment.

## Custom network models
Look at network/Network.py and network/Model.py to learn how to create your own networks. 

## How to run with custom environments and custom network models
Look at ga3c_atari.py or ga3c_cartpole.py to learn how to write your own script to train the GA3C agent to play your environment.

## References
* [Babaeizadeh, Mohammad and Frosio, Iuri and Tyree, Stephen and Clemons, Jason and Kautz, Jan. ICLR 2017, Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU](https://arxiv.org/abs/1611.06256)
* [NVlabs/GA3C](https://github.com/NVlabs/GA3C)