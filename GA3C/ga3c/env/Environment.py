from GA3C.ga3c.Config import Config
from GA3C.ga3c.env.AtariEnvironment import AtariEnvironment
from GA3C.ga3c.env.CartPoleEnvironment import CartPoleEnvironment
from GA3C.ga3c.env.NBVEnvironment import NBVEnvironment

def Environment():
	if 'nbv' in Config.NETWORK_NAME:
		return NBVEnvironment()
	elif 'atari' in Config.NETWORK_NAME:
		return AtariEnvironment()
	elif 'cartpole' in Config.NETWORK_NAME:
		return CartPoleEnvironment()
	else:
		raise('Env does not exist.')
