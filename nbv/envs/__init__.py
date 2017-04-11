from nbv.envs.nbv_env import NBVEnvV0
from gym.envs.registration import register

register(
    id='Next-Best-View-v0',
    entry_point='nbv.envs.nbv_env:NBVEnvV0',
    kwargs={'max_steps': 12})

register(
    id='Next-Best-View-v1',
    entry_point='nbv.envs.nbv_env:NBVEnvV1',
    kwargs={'max_steps': 12})

register(
    id='Next-Best-View-v2',
    entry_point='nbv.envs.nbv_env:NBVEnvV2',
    kwargs={'max_steps': 12})

register(
    id='Next-Best-View-v3',
    entry_point='nbv.envs.nbv_env:NBVEnvV3',
    kwargs={'max_steps': 12})