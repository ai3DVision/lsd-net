# check python version; warn if not Python3
import sys
import warnings
if sys.version_info < (3,0):
    warnings.warn("Optimized for Python3. Performance may suffer under Python2.", Warning)

import gym

from GA3C.ga3c.Config import Config
from GA3C.ga3c.Server import Server
from GA3C.ga3c.env.Environment import Environment
from GA3C.ga3c.network.Network import Network

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# Parse arguments
for i in range(1, len(sys.argv)):
    # Config arguments should be in format of Config=Value
    # For setting booleans to False use Config=
    x, y = sys.argv[i].split('=')
    setattr(Config, x, type(getattr(Config, x))(y))

Config.NETWORK_NAME = 'nbv'
Config.GAME = 'Next-Best-View-v0'

Config.STACKED_FRAMES = 3
Config.IMAGE_WIDTH = 224
Config.IMAGE_HEIGHT = 224

if Config.TRAIN_MODELS:
	Config.PLAY_MODE = False
	Config.AGENTS = 8
	# Config.PREDICTORS = 1
	# Config.TRAINERS = 1
	# Config.DYNAMIC_SETTINGS = False

	Config.EPISODES = 200000

	Config.TENSORBOARD = True
	Config.TENSORBOARD_UPDATE_FREQUENCY = 100
	Config.SAVE_FREQUENCY = 2500

	Config.GREEDY_POLICY = False
	Config.LINEAR_DECAY_GREEDY_EPSILON_POLICY = True
	Config.EPSILON_START = 1
	Config.EPSILON_END = 0.1
	Config.DECAY_NUM_STEPS = 400000

	Config.LEARNING_RATE_START = 0.00001
	Config.LEARNING_RATE_END = 0.00001

	gym.undo_logger_setup()

	# Start main program
	Server().main()
else:
	env = gym.make(Config.GAME)

	network = Network(Config.DEVICE, Config.NETWORK_NAME, env.action_space.n)
	network.load()

	env.test_ga3c(network, num_episode=1)
