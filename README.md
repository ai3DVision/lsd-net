# Next Best View

## Requirements
* python 3.4.2
* keras 2.0.2
* tensorflow 1.0.1
* gym 0.8.1
* pillow 4.0.0

## Setup the NBV env
Download the modelnet40v1 data from http://maxwell.cs.umass.edu/mvcnn-data/ and put the modelnet40v1 folder inside the nbv/envs/ folder

cd nbv/envs/
wget http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1.tar
tar -xvf modelnet40v1.tar
rm modelnet40v1.tar

## Test the NBV env
```
python test_env.py
```

## Train the DQN on NBV
```
python dqn_nbv.py
```

## Test the DQN
```
python dqn_nbv.py --phase test --dir {WEIGHT DIRECTORY}
```

## Train the GA3C on NBV
```
python ga3c_nbv.py
```

## Test the GA3C on NBV
```
python ga3c_nbv.py PLAY_MODE=1
```

## References
* [Wu, Zhirong and Song, Shuran and Khosla, Aditya and Tang, Xiaoou  and Xiao, Jianxiong, CVPR 2015, 3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/abs/1406.5670)
* [Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin A., CoRR 2013, Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [Babaeizadeh, Mohammad and Frosio, Iuri and Tyree, Stephen and Clemons, Jason and Kautz, Jan. ICLR 2017, Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU](https://arxiv.org/abs/1611.06256)
* [NVlabs/GA3C](https://github.com/NVlabs/GA3C)
* [Tensorflow Models](https://github.com/tensorflow/models)
* [Dataset](http://maxwell.cs.umass.edu/mvcnn-data/)