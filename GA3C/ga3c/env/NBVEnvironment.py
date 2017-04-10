from GA3C.ga3c.Config import Config
from GA3C.ga3c.GameManager import GameManager
import numpy as np
from nbv.envs import NBVEnvV0

class NBVEnvironment():
    def __init__(self):
        self.game = GameManager(Config.GAME, display=Config.PLAY_MODE)
        self.previous_state = None
        self.current_state = None
        self.reset()

    def get_num_actions(self):
        return self.game.env.action_space.n

    def reset(self):
        self.previous_state = None
        self.current_state = None
        self.game.reset()

    def step(self, action):
        observation, reward, done, _ = self.game.step(action)
        self.previous_state = self.current_state
        self.current_state = observation
        return reward, done