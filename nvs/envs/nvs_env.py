import os
import ast
import random
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from nvs.envs.env_constants import data_folder, data_dict_file_name

class NVSEnv():
	metadata = {'render.modes': ['human']}

	dir_path = os.path.dirname(os.path.realpath(__file__))
	data_folder = os.path.join(dir_path, 'data')
	image_folder = os.path.join(dir_path, 'modelnet40v1')

	category = None
	group = None
	image_idx = None
	group_size = None
	image = None
	render_delay = 1

	def __init__(self):
		self.data = self.read_data()

		self.actions = {}
		self.nS = 0

		for category in self.data['train']:
			for group in self.data['train'][category]:
				self.nS = self.nS + self.data['train'][category][group]['size']
				label = self.data['train'][category][group]['label']
				self.actions[label] = category

		self.actions[len(self.actions)] = 'CW' # clockwise
		self.actions[len(self.actions)] = 'CCW' # counter clockwise

		self.nA = len(self.actions)

	def reset(self):
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
		if self.actions[action] == 'CCW':
			self.image_idx = (self.image_idx - 1) % self.group_size
			self.image = self.get_current_image()
			obs, reward, is_terminal, info = self.image, 0, 0, {}
		elif self.actions[action] == 'CW':
			self.image_idx = (self.image_idx + 1) % self.group_size
			self.image = self.get_current_image()
			obs, reward, is_terminal, info = self.image, 0, 0, {}
		elif self.actions[action] == self.category:
			obs, reward, is_terminal, info = self.image, 1, 1, {}
		else:
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
		np.random.seed(seed)

	def read_data(self):
		full_data_dict_file_name = os.path.join(self.data_folder, data_dict_file_name)
		data = [line for line in open(full_data_dict_file_name, 'r')][0]
		data = ast.literal_eval(data)
		return data

	def get_current_image(self):
		image_path = self.data['train'][self.category][self.group]['images'][self.image_idx]
		image = np.array(Image.open(image_path))
		return image