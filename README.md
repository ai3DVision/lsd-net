# Next Best View

## Requirements
* python 3.4.2
* keras 2.0.2
* tensorflow  1.0.1
* gym 0.8.1

## Setup the NBV env
Download the modelnet40v1 data from http://maxwell.cs.umass.edu/mvcnn-data/ and put the modelnet40v1 folder inside the nbv/envs/ folder

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