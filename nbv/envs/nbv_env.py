import os
import ast
import random
from PIL import Image
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import time
from gym import Env, spaces
import sys

from nbv.envs.env_constants import data_folder, data_dict_file_name, \
								   output_folder_name

class NBVEnvV0(Env):
	metadata = {'render.modes': ['human']}

	# Path to data and images
	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_folder = os.path.join(dir_path, output_folder_name)

	# Env state
	category = None
	group = None
	image_idx = None
	group_size = None
	image = None
	steps = 0

	# Time delay during render
	render_delay = 1

	def __init__(self, max_steps):
		# Get dictionary of data
		self.data = self.read_data()

		# Get actions and count nS from data
		self.actions = {}
		self.nS = 0
		for category in self.data['train']:
			for group in self.data['train'][category]:
				self.nS = self.nS + self.data['train'][category][group]['size']
				label = self.data['train'][category][group]['label']
				self.actions[label] = category

		# Add turn clockwise and turn counter clockwise actions
		self.actions[len(self.actions)] = 'CW'
		self.actions[len(self.actions)] = 'CCW'

		self.nA = len(self.actions)
		self.action_space = spaces.Discrete(self.nA)

		self.max_steps = max_steps

	def reset(self):
		# Get a random image and store the state
		categories = list(self.data['train'].keys())
		category = random.choice(categories)

		groups = list(self.data['train'][category].keys())
		group = random.choice(groups)

		group_size = self.data['train'][category][group]['size']
		image_idx = random.randint(1, group_size) - 1

		self.category = category
		self.group = group
		self.image_idx = image_idx
		self.group_size = group_size
		self.image = self.get_current_image()
		self.steps = 0

		return self.image

	def step(self, action):
		if self.actions[action] == 'CW':
			# Increment image index by 1 to turn clockwise
			# Ex: airplane_0627_001.jpg -> airplane_0627_002.jpg
			self.image_idx = (self.image_idx + 1) % self.group_size
			self.image = self.get_current_image()
			obs, reward, is_terminal, info = self.image, 0, 0, {}
		elif self.actions[action] == 'CCW':
			# Decrement image index by 1 to turn counter clockwise
			# Ex: airplane_0627_002.jpg -> airplane_0627_001.jpg
			self.image_idx = (self.image_idx - 1) % self.group_size
			self.image = self.get_current_image()
			obs, reward, is_terminal, info = self.image, 0, 0, {}
		elif self.actions[action] == self.category:
			# Classify correctly
			obs, reward, is_terminal, info = self.image, 1, 1, {}
		else:
			# Classify incorrectly
			obs, reward, is_terminal, info = self.image, 0, 0, {}

		self.steps = self.steps + 1

		# Terminate if reached max number of steps
		if self.steps >= self.max_steps:
			is_terminal = 1

		return obs, reward, is_terminal, info

	def render(self, mode='human', close=False):
		# plt.imshow(self.image)
		# plt.show(block=False)
		# time.sleep(self.render_delay)
		# plt.close()
		pass
		
	def close(self):
		pass

	def set_render_delay(self, render_delay):
		self.render_delay = render_delay

	def seed(self, seed=None):
		random.seed(seed)

	def read_data(self):
		# Read dictionary of data from file created by format_data.py
		full_data_dict_file_name = os.path.join(self.data_folder, data_dict_file_name)
		data = [line for line in open(full_data_dict_file_name, 'r')][0]
		data = ast.literal_eval(data)
		return data

	def get_current_image(self):
		# Get RBG image
		image_path = self.data['train'][self.category][self.group]['images'][self.image_idx]
		image_path = os.path.join(self.dir_path, image_path)
		image = np.array(Image.open(image_path))
		return image

	def test_dqn(self, dqnAgent, num_episode, data_type='test'):
		num_correct = 0
		total_groups = 0
		accuracies = []

		for _ in range(num_episode):
			for category in self.data[data_type]:
				for group in self.data[data_type][category]:
					total_groups = total_groups + 1
					group_size = self.data[data_type][category][group]['size']
					image_idx = random.randint(1, group_size) - 1

					# Max steps is self.max_steps
					for _ in range(self.max_steps):
						image_path = self.data[data_type][category][group]['images'][image_idx]
						image_path = os.path.join(self.dir_path, image_path)
						state = np.array(Image.open(image_path))

						state = np.array([state])
						action, _ = dqnAgent.select_action(state)
						action = action[0]

						if self.actions[action] == 'CW':
							image_idx = (image_idx + 1) % group_size
						elif self.actions[action] == 'CCW':
							image_idx = (image_idx - 1) % group_size
						elif self.actions[action] == category:
							num_correct = num_correct + 1
							break
						else:
							break
			
			accuracies.append(num_correct / float(total_groups))

		print('The accuracy is %f +/- %f' % (np.mean(accuracies), np.std(accuracies)))
		sys.stdout.flush()

	def test_ga3c(self, network, num_episode, data_type='test'):
		num_correct = 0
		total_groups = 0
		accuracies = []

		for _ in range(num_episode):
			for category in self.data[data_type]:
				for group in self.data[data_type][category]:
					total_groups = total_groups + 1
					group_size = self.data[data_type][category][group]['size']
					image_idx = random.randint(1, group_size) - 1

					# Max steps is self.max_steps
					for _ in range(self.max_steps):
						image_path = self.data[data_type][category][group]['images'][image_idx]
						image_path = os.path.join(self.dir_path, image_path)
						state = np.array(Image.open(image_path))

						state = np.array([state]) / 255.
						p, v = network.predict_p_and_v(state)
						action = np.argmax(p[0])
						
						if self.actions[action] == 'CW':
							image_idx = (image_idx + 1) % group_size
						elif self.actions[action] == 'CCW':
							image_idx = (image_idx - 1) % group_size
						elif self.actions[action] == category:
							num_correct = num_correct + 1
							break
						else:
							break

			accuracies.append(num_correct / float(total_groups))

		print('The accuracy is %f +/- %f' % (np.mean(accuracies), np.std(accuracies)))
		sys.stdout.flush()

class NBVEnvV1(NBVEnvV0):
	def __init__(self, max_steps):
		NBVEnvV0.__init__(self, max_steps)

	def step(self, action):
		obs, reward, is_terminal, info = NBVEnvV0.step(self, action)
		if self.steps < self.max_steps:
			is_terminal = 0
		return obs, reward, is_terminal, info

class NBVEnvV2(NBVEnvV0):
	def __init__(self, max_steps):
		NBVEnvV0.__init__(self, max_steps)

	def step(self, action):
		obs, reward, is_terminal, info = NBVEnvV0.step(self, action)
		if self.actions[action] != 'CW' \
		   and self.actions[action] != 'CCW' \
		   and self.actions[action] != self.category:
			reward = -1
		return obs, reward, is_terminal, info

class NBVEnvV3(NBVEnvV1):
	def __init__(self, max_steps):
		NBVEnvV1.__init__(self, max_steps)

	def step(self, action):
		obs, reward, is_terminal, info = NBVEnvV1.step(self, action)
		if self.actions[action] != 'CW' \
		   and self.actions[action] != 'CCW' \
		   and self.actions[action] != self.category:
			reward = -1
		return obs, reward, is_terminal, info