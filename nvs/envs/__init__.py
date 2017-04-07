from nvs.envs.nvs_env import NVSEnvV0
from gym.envs.registration import register

register(
    id='New-View-Synthesis-v0',
    entry_point='nvs.envs.nvs_env:NVSEnvV0',
    kwargs={'max_steps': 12})