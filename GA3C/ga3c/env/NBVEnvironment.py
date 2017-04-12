from GA3C.ga3c.Config import Config
from GA3C.ga3c.GameManager import GameManager
import numpy as np
from nbv.envs import NBVEnvV0
from PIL import Image

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
        state = self.game.reset()
        state = NBVEnvironment._preprocess(state)
        self.current_state = state

    def step(self, action):
        state, reward, done, _ = self.game.step(action)
        state = NBVEnvironment._preprocess(state)
        self.previous_state = self.current_state
        self.current_state = state
        return reward, done

    @staticmethod
    def _preprocess(state):
        img = Image.fromarray(state)
        img = img.resize((Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        state = np.array(img)
        state = state.astype('float32') / 255.
        return state