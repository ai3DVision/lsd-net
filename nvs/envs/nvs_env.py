import os
import ast
import random
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from gym import Env, spaces

from nvs.envs.env_constants import data_folder, data_dict_file_name, \
								   output_folder_name

class NVSEnv(Env):
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

	# Time delay during render
	render_delay = 1

	def __init__(self):
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

	def reset(self):
		# Get a random image and store the state
		categories = list(self.data['train'].keys())
		category = random.choice(categories)

		groups = list(self.data['train'][category].keys())
		group = random.choice(groups)

		size = self.data['train'][category][group]['size']
		image_idx = random.randint(1, 10)
		
		self.category = category
		self.group = group
		self.image_idx = image_idx
		self.group_size = size
		self.image = self.get_current_image()

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

		return obs, reward, is_terminal, info

	def render(self, mode='human', close=False):
		plt.imshow(self.image)
		plt.show(block=False)
		time.sleep(self.render_delay)
		plt.close()

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