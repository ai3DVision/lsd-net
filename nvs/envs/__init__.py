from nvs.envs.nvs_env import NVSEnv
from gym.envs.registration import register

register(
    id='New-View-Synthesis-v0',
    entry_point='nvs.envs.nvs_env:NVSEnv',
    kwargs={'max_steps': 12})