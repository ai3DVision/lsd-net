from nvs.envs.nvs_env import NVSEnv
from gym.envs.registration import register
from nvs.envs import env_constants

register(
    id='New-View-Synthesis-v0',
    entry_point='nvs.nvs_env:NVSEnv')