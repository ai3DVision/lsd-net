from GA3C.ga3c.Config import Config
from GA3C.ga3c.GameManager import GameManager
import numpy as np
from nbv.envs import NBVEnvV0
from PIL import Image
import sys
if sys.version_info[0] == 2:
    from env import Environment
else:
    from A3C.a3c.env.env import Environment

class NBVEnvironment(Environment):
    def __init__(self, gym_env, resized_width, resized_height, agent_history_length):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.gym_actions = range(gym_env.action_space.n)
        self.reset()
    
    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        state = NBVEnvironment._preprocess(state, self.resized_width, self.resized_height)
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = NBVEnvironment._preprocess(state, self.resized_width, self.resized_height)
        return state, reward, done, info

    @staticmethod
    def _preprocess(state, resized_width, resized_height):
        img = Image.fromarray(state)
        img = img.resize((resized_width, resized_height))
        state = np.array(img)
        state = state.astype('float32') / 255.
        return state
