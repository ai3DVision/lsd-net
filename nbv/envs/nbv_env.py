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

import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc

from GA3C.ga3c.Config import Config

class NBVEnvV0(Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, max_steps):
		# Path to data and images
		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		self.data_folder = os.path.join(self.dir_path, output_folder_name)

		# Env state
		self.category = None
		self.group = None
		self.image_idx = None
		self.group_size = None
		self.image = None
		self.steps = 0

		# Time delay during render
		render_delay = 1

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
		accuracies = []
		movements = []
		max_steps_instances = []
		max_steps_object_instances = []
		object_movements = {}

		for i in range(num_episode):
			print('Testing episode %d' % i)
			num_correct = 0
			total_groups = 0
			move_count = 0
			max_steps_instance = 0
			for category in self.data[data_type]:
				for group in self.data[data_type][category]:
					total_groups = total_groups + 1
					group_size = self.data[data_type][category][group]['size']
					image_idx = random.randint(1, group_size) - 1

					# Max steps is self.max_steps
					took_max_steps = False
					for j in range(self.max_steps):
						image_path = self.data[data_type][category][group]['images'][image_idx]
						image_path = os.path.join(self.dir_path, image_path)
						state = np.array(Image.open(image_path))

						state = np.array([state])
						action, q_values = dqnAgent.select_action(state)
						action = action[0]

						if j == self.max_steps-1:
							q_values = q_values[0]
							q_values = q_values[0:-2]
							action = dqnAgent.policy.select_action(q_values)
							took_max_steps = True

						if self.actions[action] == 'CW':
							prev_image_idx = image_idx
							image_idx = (image_idx + 1) % group_size
							move_count = move_count + 1
							if category not in object_movements:
								object_movements[category] = {}
							if group not in object_movements[category]:
								object_movements[category][group] = []
							object_movements[category][group].append((prev_image_idx, image_idx))
						elif self.actions[action] == 'CCW':
							prev_image_idx = image_idx
							image_idx = (image_idx - 1) % group_size
							move_count = move_count + 1
							if category not in object_movements:
								object_movements[category] = {}
							if group not in object_movements[category]:
								object_movements[category][group] = []
							object_movements[category][group].append((prev_image_idx, image_idx))
						elif self.actions[action] == category:
							num_correct = num_correct + 1
							break
						else:
							break

					if took_max_steps:
						max_steps_instance = max_steps_instance + 1
						max_steps_object_instances.append(group)

			accuracy = num_correct / float(total_groups)
			print('Accuracy: %f' % accuracy)
			print('Movement: %d' % move_count)
			print('Number of objects that took max steps: %d' % max_steps_instance)
			accuracies.append(accuracy)
			movements.append(move_count)
			max_steps_instances.append(max_steps_instance)
			sys.stdout.flush()
			
		print('The accuracy is %f +/- %f' % (np.mean(accuracies), np.std(accuracies)))
		print('The movement is %f +/- %f' % (np.mean(movements), np.std(movements)))
		print('The number of objects that took the max number of steps is %f +/- %f' % (np.mean(max_steps_instances), np.std(max_steps_instances)))
		print('The object movements are %s' % str(object_movements))
		print('The objects that took the max number of steps is %s' % str(max_steps_object_instances))
		sys.stdout.flush()

	def test_ga3c(self, network, num_episode, data_type='test'):
		accuracies = []

		nb_frames = Config.STACKED_FRAMES
		frame_q = Queue(maxsize=nb_frames)
		movements = []
		max_steps_instances = []
		max_steps_object_instances = []
		object_movements = {}

		for i in range(num_episode):
			print('Testing episode %d' % i)
			num_correct = 0
			total_groups = 0
			move_count = 0
			max_steps_instance = 0
			for category in self.data[data_type]:
				for group in self.data[data_type][category]:
					total_groups = total_groups + 1
					group_size = self.data[data_type][category][group]['size']
					image_idx = random.randint(1, group_size) - 1

					# Max steps is self.max_steps
					took_max_steps = False
					for j in range(self.max_steps):
						image_path = self.data[data_type][category][group]['images'][image_idx]
						image_path = os.path.join(self.dir_path, image_path)
						image = np.array(Image.open(image_path))

						image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
						image = misc.imresize(image, [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], 'bilinear')
						image = image.astype(np.float32) / 128.0 - 1.0
						frame_q.put(image)
						
						while not frame_q.full():
							frame_q.put(image)

						x_ = np.array(frame_q.queue)
						x_ = np.transpose(x_, [1, 2, 0])

						p, v = network.predict_p_and_v(np.array([x_]))
						action = np.argmax(p[0])
						
						if j == self.max_steps-1:
							p = p[0]
							p = p[0:-2]
							action = np.argmax(p)
							took_max_steps = True
							
						if self.actions[action] == 'CW':
							prev_image_idx = image_idx
							image_idx = (image_idx + 1) % group_size
							move_count = move_count + 1
							if category not in object_movements:
								object_movements[category] = {}
							if group not in object_movements[category]:
								object_movements[category][group] = []
							object_movements[category][group].append((prev_image_idx, image_idx))
						elif self.actions[action] == 'CCW':
							prev_image_idx = image_idx
							image_idx = (image_idx - 1) % group_size
							move_count = move_count + 1
							if category not in object_movements:
								object_movements[category] = {}
							if group not in object_movements[category]:
								object_movements[category][group] = []
							object_movements[category][group].append((prev_image_idx, image_idx))
						elif self.actions[action] == category:
							num_correct = num_correct + 1
							break
						else:
							break

					frame_q.queue.clear()
					
					if took_max_steps:
						max_steps_instance = max_steps_instance + 1
						max_steps_object_instances.append(group)

			accuracy = num_correct / float(total_groups)
			print('Accuracy: %f' % accuracy)
			print('Movement: %d' % move_count)
			print('Number of objects that took max steps: %d' % max_steps_instance)
			accuracies.append(accuracy)
			movements.append(move_count)
			max_steps_instances.append(max_steps_instance)
			sys.stdout.flush()

		print('The accuracy is %f +/- %f' % (np.mean(accuracies), np.std(accuracies)))
		print('The movement is %f +/- %f' % (np.mean(movements), np.std(movements)))
		print('The number of objects that took the max number of steps is %f +/- %f' % (np.mean(max_steps_instances), np.std(max_steps_instances)))
		print('The object movements are %s' % str(object_movements))
		print('The objects that took the max number of steps is %s' % str(max_steps_object_instances))
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

class NBVEnvV4(NBVEnvV0):
	def __init__(self, max_steps):
		NBVEnvV0.__init__(self, max_steps)

	def step(self, action):
		obs, reward, is_terminal, info = NBVEnvV0.step(self, action)
		if self.actions[action] != 'CW' and self.actions[action] != 'CCW':
			is_terminal = True
		return obs, reward, is_terminal, info

class NBVEnvV5(NBVEnvV0):
	def __init__(self, max_steps):
		NBVEnvV0.__init__(self, max_steps)

	def step(self, action):
		obs, reward, is_terminal, info = NBVEnvV0.step(self, action)
		if self.actions[action] != 'CW' and self.actions[action] != 'CCW':
			is_terminal = True
		if self.actions[action] != 'CW' \
		   and self.actions[action] != 'CCW' \
		   and self.actions[action] != self.category:
			reward = -1
		return obs, reward, is_terminal, info