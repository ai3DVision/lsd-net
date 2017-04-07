from nbv.envs.nbv_env import NBVEnvV0
from gym.envs.registration import register

register(
    id='Next-Best-View-v0',
    entry_point='nbv.envs.nbv_env:NBVEnvV0',
    kwargs={'max_steps': 12})