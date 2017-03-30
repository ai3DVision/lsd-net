import numpy as np
import tensorflow as tf
from deeprl_hw2.core import Sample, Observation
from deeprl_hw2.objectives import huber_loss, mean_huber_loss
from deeprl_hw2.ringbuffer import RingBuffer
from deeprl_hw2.memory import NaiveMemory, BasicMemory
import inspect
from deeprl_hw2.preprocessors import HistoryPreprocessor
from deeprl_hw2.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy
from dqn_atari import create_model
import argparse
import gym
from deeprl_hw2.preprocessors import AtariPreprocessor
from deeprl_hw2.dqn import DQNAgent

def testHuberLoss():
	with tf.Session() as sess:
		y_true = tf.constant([0.7,0.3,0.8,0.1])
		y_pred = tf.constant([2.1,0.4,0.9,3.2])

		loss = sess.run(huber_loss(y_true, y_pred))
		mean_loss = sess.run(mean_huber_loss(y_true, y_pred))

		assert(np.isclose(loss, [0.9, 0.005, 0.005, 2.6]).all())
		assert(np.isclose(mean_loss, 0.8775))

def testRingBuffer():
	rb = RingBuffer(5)
	rb.append(1)
	rb.append(2)
	rb.append(3)

	assert(rb.buffer == [1,2,3,None,None])

	rb.append(1)
	rb.append(2)
	rb.append(3)

	assert(rb.buffer == [3,2,3,1,2])
	
	rb.append(1)
	rb.append(2)
	rb.append(3)
	rb.append(4)
	rb.append(5)

	assert(rb.buffer == [5,1,2,3,4])

def testBasicMemory():
	bm = BasicMemory(10, 3)

	bm.append(np.array([[0,0],[0,0]]), 0, 1, False)
	bm.append(np.array([[1,1],[1,1]]), 1, 1, False)
	bm.append(np.array([[2,2],[2,2]]), 2, 1, False)
	bm.append(np.array([[3,3],[3,3]]), 3, 1, True)
	bm.append(np.array([[4,4],[4,4]]), 0, 1, False)

	assert(bm.sample(3, indexes=[0, -4, -3]) == [Sample(np.array([[[ 0.,  0.,  0.], \
																   [ 0.,  0.,  0.]], \
																  [[ 0.,  0.,  0.], \
																   [ 0.,  0.,  0.]]]), 0, 1, \
														np.array([[[ 0.,  0.,  1.], \
																   [ 0.,  0.,  1.]], \
																  [[ 0.,  0.,  1.], \
																   [ 0.,  0.,  1.]]]), False),
												 Sample(np.array([[[ 0.,  0.,  1.], \
																   [ 0.,  0.,  1.]], \
																  [[ 0.,  0.,  1.], \
																   [ 0.,  0.,  1.]]]), 1, 1, \
														np.array([[[ 0.,  1.,  2.], \
																   [ 0.,  1.,  2.]], \
																  [[ 0.,  1.,  2.], \
																   [ 0.,  1.,  2.]]]), False),
												 Sample(np.array([[[ 0.,  1.,  2.], \
																   [ 0.,  1.,  2.]], \
																  [[ 0.,  1.,  2.], \
																   [ 0.,  1.,  2.]]]), 2, 1, \
														np.array([[[ 1.,  2.,  3.], \
																   [ 1.,  2.,  3.]], \
																  [[ 1.,  2.,  3.], \
																   [ 1.,  2.,  3.]]]), False)])

	bm.append(np.array([[5,5],[5,5]]), 1, 1, False)
	bm.append(np.array([[6,6],[6,6]]), 2, 1, True)
	bm.append(np.array([[7,7],[7,7]]), 3, 1, False)
	bm.append(np.array([[8,8],[8,8]]), 0, 1, False)
	bm.append(np.array([[9,9],[9,9]]), 1, 1, False)
	bm.append(np.array([[10,10],[10,10]]), 2, 1, False)
	bm.append(np.array([[11,11],[11,11]]), 3, 1, False)
	bm.append(np.array([[12,12],[12,12]]), 0, 1, False)

	assert(bm.memory.buffer == [Observation(np.array(10), 2, 1, False), 
		                        Observation(np.array(11), 3, 1, False), 
		                        Observation(np.array(12), 0, 1, False), 
		                        Observation(np.array(3), 3, 1, True), 
		                        Observation(np.array(4), 0, 1, False), 
		                        Observation(np.array(5), 1, 1, False), 
		                        Observation(np.array(6), 2, 1, True), 
		                        Observation(np.array(7), 3, 1, False), 
		                        Observation(np.array(8), 0, 1, False), 
		                        Observation(np.array(9), 1, 1, False)])
	
	assert(bm.sample(5, indexes=[0, 4, 5, 8, 9]) == [Sample(np.array([[[ 8.,  9.,  10.], \
																	   [ 8.,  9.,  10.]], \
																	  [[ 8.,  9.,  10.], \
																	   [ 8.,  9.,  10.]]]), 2, 1, \
															np.array([[[ 9.,  10.,  11.], \
																	   [ 9.,  10.,  11.]], \
																	  [[ 9.,  10.,  11.], \
																	   [ 9.,  10.,  11.]]]), False), 
													 Sample(np.array([[[ 0.,  0.,  4.], \
																	   [ 0.,  0.,  4.]], \
																	  [[ 0.,  0.,  4.], \
																	   [ 0.,  0.,  4.]]]), 0, 1, \
															np.array([[[ 0.,  4.,  5.], \
																	   [ 0.,  4.,  5.]], \
																	  [[ 0.,  4.,  5.], \
																	   [ 0.,  4.,  5.]]]), False), 
													 Sample(np.array([[[ 0.,  4.,  5.], \
																	   [ 0.,  4.,  5.]], \
																	  [[ 0.,  4.,  5.], \
																	   [ 0.,  4.,  5.]]]), 1, 1, \
															np.array([[[ 4.,  5.,  6.], \
																	   [ 4.,  5.,  6.]], \
																	  [[ 4.,  5.,  6.], \
																	   [ 4.,  5.,  6.]]]), False), 
													 Sample(np.array([[[ 0.,  7.,  8.], \
																	   [ 0.,  7.,  8.]], \
																	  [[ 0.,  7.,  8.], \
																	   [ 0.,  7.,  8.]]]), 0, 1, \
															np.array([[[ 7.,  8.,  9.], \
																	   [ 7.,  8.,  9.]], \
																	  [[ 7.,  8.,  9.], \
																	   [ 7.,  8.,  9.]]]), False), 
													 Sample(np.array([[[ 7.,  8.,  9.], \
																	   [ 7.,  8.,  9.]], \
																	  [[ 7.,  8.,  9.], \
																	   [ 7.,  8.,  9.]]]), 1, 1, \
															np.array([[[ 8.,  9.,  10.], \
																	   [ 8.,  9.,  10.]], \
																	  [[ 8.,  9.,  10.], \
																	   [ 8.,  9.,  10.]]]), False)])

	bm.clear()
	assert(bm.memory.buffer == 10 * [None])
	
	bm = BasicMemory(10, 1)

	bm.append(np.array([0]), 1, 0, False)
	bm.append(np.array([4]), 1, 0, False)
	bm.append(np.array([8]), 2, 0, False)
	bm.append(np.array([9]), 1, 0, False)
	bm.append(np.array([13]), 2, 0, False)
	bm.append(np.array([14]), 2, 1, True)
	bm.append(np.array([15]), 0, 0, True)
	bm.append(np.array([0]), 2, 0, False)
	bm.append(np.array([1]), 2, 0, False)
	bm.append(np.array([2]), 1, 0, False)
	assert(bm.sample(8, indexes=[0,1,2,3,4,5,7,8]) == [Sample(np.array([[ 0.]]), 1, 0, np.array([[ 4.]]), False), \
													   Sample(np.array([[ 4.]]), 1, 0, np.array([[ 8.]]), False), \
													   Sample(np.array([[ 8.]]), 2, 0, np.array([[ 9.]]), False), \
													   Sample(np.array([[ 9.]]), 1, 0, np.array([[ 13.]]), False), \
													   Sample(np.array([[ 13.]]), 2, 0, np.array([[ 14.]]), False), \
													   Sample(np.array([[ 14.]]), 2, 1, np.array([[ 15.]]), True), \
													   Sample(np.array([[ 0.]]), 2, 0, np.array([[ 1.]]), False), \
													   Sample(np.array([[ 1.]]), 2, 0, np.array([[ 2.]]), False)])

def testNaiveMemory():
	nm = NaiveMemory(10, 3)

	nm.append(np.array([[0,0],[0,0]]), 0, 1, False)
	nm.append(np.array([[1,1],[1,1]]), 1, 1, False)
	nm.append(np.array([[2,2],[2,2]]), 2, 1, False)
	nm.append(np.array([[3,3],[3,3]]), 3, 1, True)
	nm.append(np.array([[4,4],[4,4]]), 0, 1, False)
	nm.append(np.array([[5,5],[5,5]]), 1, 1, False)
	nm.append(np.array([[6,6],[6,6]]), 2, 1, True)
	nm.append(np.array([[7,7],[7,7]]), 3, 1, False)
	nm.append(np.array([[8,8],[8,8]]), 0, 1, False)
	nm.append(np.array([[9,9],[9,9]]), 1, 1, False)
	nm.append(np.array([[10,10],[10,10]]), 2, 1, False)
	nm.append(np.array([[11,11],[11,11]]), 3, 1, False)
	
	assert(nm.sample(3) == [Sample(np.array([[[ 0.,  7.,  8.], \
											  [ 0.,  7.,  8.]], \
											 [[ 0.,  7.,  8.], \
											  [ 0.,  7.,  8.]]]), 0, 1, \
								   np.array([[[ 7.,  8.,  9.], \
											  [ 7.,  8.,  9.]], \
											 [[ 7.,  8.,  9.], \
											  [ 7.,  8.,  9.]]]), False), 
							Sample(np.array([[[ 7.,  8.,  9.], \
											  [ 7.,  8.,  9.]], \
											 [[ 7.,  8.,  9.], \
											  [ 7.,  8.,  9.]]]), 1, 1, \
								   np.array([[[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]], \
											 [[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]]]), False), 
							Sample(np.array([[[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]], \
											 [[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]]]), 2, 1, \
								   np.array([[[ 9.,  10.,  11.], \
											  [ 9.,  10.,  11.]], \
											 [[ 9.,  10.,  11.], \
											  [ 9.,  10.,  11.]]]), False)])

	nm.append(np.array([[12,12],[12,12]]), 0, 1, False)

	assert(nm.memory.buffer == [Observation(np.array([[10,10],[10,10]]), 2, 1, False), 
		                        Observation(np.array([[11,11],[11,11]]), 3, 1, False), 
		                        Observation(np.array([[12,12],[12,12]]), 0, 1, False), 
		                        Observation(np.array([[3,3],[3,3]]), 3, 1, True), 
		                        Observation(np.array([[4,4],[4,4]]), 0, 1, False), 
		                        Observation(np.array([[5,5],[5,5]]), 1, 1, False), 
		                        Observation(np.array([[6,6],[6,6]]), 2, 1, True), 
		                        Observation(np.array([[7,7],[7,7]]), 3, 1, False), 
		                        Observation(np.array([[8,8],[8,8]]), 0, 1, False), 
		                        Observation(np.array([[9,9],[9,9]]), 1, 1, False)])

	assert(nm.sample(3) == [Sample(np.array([[[ 7.,  8.,  9.], \
											  [ 7.,  8.,  9.]], \
											 [[ 7.,  8.,  9.], \
											  [ 7.,  8.,  9.]]]), 1, 1, \
								   np.array([[[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]], \
											 [[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]]]), False), 
							Sample(np.array([[[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]], \
											 [[ 8.,  9.,  10.], \
											  [ 8.,  9.,  10.]]]), 2, 1, \
								   np.array([[[ 9.,  10.,  11.], \
											  [ 9.,  10.,  11.]], \
											 [[ 9.,  10.,  11.], \
											  [ 9.,  10.,  11.]]]), False), 
							Sample(np.array([[[ 9.,  10.,  11.], \
											  [ 9.,  10.,  11.]], \
											 [[ 9.,  10.,  11.], \
											  [ 9.,  10.,  11.]]]), 3, 1, \
								   np.array([[[ 10.,  11.,  12.], \
											  [ 10.,  11.,  12.]], \
											 [[ 10.,  11.,  12.], \
											  [ 10.,  11.,  12.]]]), False), ])

	nm.clear()
	assert(nm.memory.buffer == 10 * [None])

def testHistoryPreprocessor():
	a = np.array([[1,1],[1,1]])
	b = np.array([[2,2],[2,2]])
	c = np.array([[3,3],[3,3]])
	d = np.array([[4,4],[4,4]])
	e = np.array([[5,5],[5,5]])

	hp = HistoryPreprocessor(a.shape, 3)

	history = np.array([[[0.,0.,0.], \
						 [0.,0.,0.]], \
					    [[0.,0.,0.], \
					     [0.,0.,0.]]])
	assert(np.array_equal(hp.history, history))

	history = np.array([[[0.,0.,1.], \
						 [0.,0.,1.]], \
					    [[0.,0.,1.], \
					     [0.,0.,1.]]])
	assert(np.array_equal(hp.process_state_for_network(a), history))

	history = np.array([[[0.,1.,2.], \
						 [0.,1.,2.]], \
						[[0.,1.,2.], \
						 [0.,1.,2.]]])
	assert(np.array_equal(hp.process_state_for_network(b), history))

	history = np.array([[[1.,2.,3.], \
						 [1.,2.,3.]], \
						[[1.,2.,3.], \
						 [1.,2.,3.]]])
	assert(np.array_equal(hp.process_state_for_network(c), history))

	history = np.array([[[2.,3.,4.], \
						 [2.,3.,4.]], \
						[[2.,3.,4.], \
						 [2.,3.,4.]]])
	assert(np.array_equal(hp.process_state_for_network(d), history))

	history = np.array([[[3.,4.,5.], \
						 [3.,4.,5.]], \
						[[3.,4.,5.], \
						 [3.,4.,5.]]])
	assert(np.array_equal(hp.process_state_for_network(e), history))

def testPolicy():
	num_actions = 6
	start_value = 1
	end_value = 0.1
	num_steps = 1000
	
	policy1 = GreedyPolicy()
	policy2 = LinearDecayGreedyEpsilonPolicy(num_actions, start_value, end_value, num_steps)

	q_values = np.array([1.0, 1.3, 1.2, 1.5, 1.1, 1.4])
	assert(policy1.select_action(q_values) == 3)

	assert(policy2.epsilon == 1)

	policy2.select_action(q_values)
	assert(np.isclose(policy2.epsilon, 0.9991))

	policy2.select_action(q_values)
	policy2.select_action(q_values)

	assert(np.isclose(policy2.epsilon, 0.9973))

	for i in range(num_steps):
		policy2.select_action(q_values)

	assert(np.isclose(policy2.epsilon, end_value))

	policy2.select_action(q_values)

	assert(np.isclose(policy2.epsilon, end_value))

def testAgent():
	parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
	parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
	parser.add_argument(
		'-o', '--output', default='atari-v0', help='Directory to save data to')
	parser.add_argument('--seed', default=0, type=int, help='Random seed')
	parser.add_argument('--input_shape', default=(84,84), type=int, help='Input shape')
	parser.add_argument('--phase', default='train', type=str, help='Train/Test/Video')
	parser.add_argument('-r', '--render', action='store_true', default=False, help='Render')
	parser.add_argument('--model', default='deep_Q_network', type=str, help='Type of model')
	parser.add_argument('-c', action='store_false', default=True, help='Cancel')
	parser.add_argument('-d', '--dir', default='', type=str, help='Directory')
	parser.add_argument('-n', '--number', default='', type=str, help='Model number')

	args = parser.parse_args()

	assert(args.phase in ['train', 'test', 'video'])
	assert(args.dir if args.phase == 'test' or args.phase == 'video' else True)

	args.input_shape = tuple(args.input_shape)

	# create the environment
	env = gym.make(args.env)

	# Number of training iterations
	num_iterations = 5000000

	# Learning rate
	alpha = 0.0001

	# Epsilion for GreedyEpsilonPolicy
	epsilon = 0.05

	# Parameters for LinearDecayGreedyEpsilonPolicy
	start_value = 0.3
	end_value = 0.05
	num_steps = 10000

	# Number of frames in the sequence
	window = 4

	# Use experience replay
	experience_replay = args.c

	# Use target fixing
	target_fixing = args.c

	# Evaluate number of episode (given the model number)
	num_episode = 1

	# DQNAgent parameters
	num_actions = env.action_space.n
	q_network = create_model(window, 
							 args.input_shape, 
							 num_actions, 
							 model_name=args.model)
	preprocessor = AtariPreprocessor(args.input_shape)
	policy = LinearDecayGreedyEpsilonPolicy(num_actions, start_value, end_value, num_steps)
	memory_size = 1000000
	gamma = 0.99
	target_update_freq = 100
	num_burn_in = 50
	train_freq = 4
	batch_size = 32
	video_capture_points = (num_iterations * np.array([0/3., 1/3., 2/3., 3/3.])).astype('int')
	save_network_freq = 100
	eval_train_freq = 50000
	eval_train_num_ep = 1

	if experience_replay:
		memory = BasicMemory(memory_size, window)
	else:
		memory = NaiveMemory(batch_size, window)

	dqnAgent = DQNAgent(args.model,
						q_network,
						preprocessor,
						memory,
						policy,
						gamma,
						target_update_freq,
						num_burn_in,
						train_freq,
						batch_size,
						num_actions,
						window,
						save_network_freq,
						video_capture_points,
						eval_train_freq,
						eval_train_num_ep,
						args.phase,
						target_fixing=target_fixing,
						render=args.render)

	q_values = np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.7], \
						 [1.3, 1.4, 1.5, 1.6, 1.1, 1.2], \
						 [1.2, 1.3, 1.4, 1.5, 2.2, 1.1], \
						 [1.5, 3.8, 1.1, 1.2, 1.3, 1.4], \
						 [0, 0, 0, 0.7, 0, 0]])
	is_terminal = np.array([0, 0, 1, 0, 1])
	reward = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
	target = dqnAgent.calc_target_values(q_values, is_terminal, reward)

	assert(np.array_equal(target, np.array([2.083, 2.084, 0.6, 4.462, 0.8])))

	bm = BasicMemory(10, 3)
	bm.append(np.array([[0,0],[0,0]]), 0, 1, False)
	bm.append(np.array([[1,1],[1,1]]), 1, 1, False)
	bm.append(np.array([[2,2],[2,2]]), 2, 1, False)
	bm.append(np.array([[3,3],[3,3]]), 3, 1, True)
	bm.append(np.array([[4,4],[4,4]]), 0, 1, False)
	bm.append(np.array([[5,5],[5,5]]), 1, 1, False)
	bm.append(np.array([[6,6],[6,6]]), 2, 1, True)
	bm.append(np.array([[7,7],[7,7]]), 3, 1, False)
	bm.append(np.array([[8,8],[8,8]]), 0, 1, False)
	bm.append(np.array([[9,9],[9,9]]), 1, 1, False)
	bm.append(np.array([[10,10],[10,10]]), 2, 1, False)
	bm.append(np.array([[11,11],[11,11]]), 3, 1, False)
	bm.append(np.array([[12,12],[12,12]]), 0, 1, False)

	minibatch = bm.sample(5, indexes=[0, 4, 5, 8, 9])

	state_batch, \
	action_batch, \
	reward_batch, \
	next_state_batch, \
	is_terminal_batch = dqnAgent.process_batch(minibatch)
	
	assert(np.array_equal(state_batch, np.array([[[[8.,9.,10.], \
												   [8.,9.,10.]], \
												  [[8.,9.,10.], \
												   [8.,9.,10.]]], \
												 [[[0.,0.,4.], \
												   [0.,0.,4.]], \
												  [[0.,0.,4.], \
												   [0.,0.,4.]]], \
												 [[[0.,4.,5.], \
												   [0.,4.,5.]], \
												  [[0.,4.,5.], \
												   [0.,4.,5.]]], \
												 [[[0.,7.,8.], \
												   [0.,7.,8.]], \
												  [[0.,7.,8.], \
												   [0.,7.,8.]]], \
												 [[[7.,8.,9.], \
												   [7.,8.,9.]], \
												  [[7.,8.,9.], \
												   [7.,8.,9.]]]])))
	assert(np.array_equal(action_batch, np.array([2, 0, 1, 0, 1])))
	assert(np.array_equal(reward_batch, np.array([1, 1, 1, 1, 1])))
	assert(np.array_equal(next_state_batch, np.array([[[[9.,10.,11.], \
												  		[9.,10.,11.]], \
													   [[9.,10.,11.], \
														[9.,10.,11.]]], \
													  [[[0.,4.,5.], \
														[0.,4.,5.]], \
													   [[0.,4.,5.], \
														[0.,4.,5.]]], \
													  [[[4.,5.,6.], \
														[4.,5.,6.]], \
													   [[4.,5.,6.], \
														[4.,5.,6.]]], \
													  [[[7.,8.,9.], \
														[7.,8.,9.]], \
													   [[7.,8.,9.], \
														[7.,8.,9.]]], \
													  [[[8.,9.,10.], \
														[8.,9.,10.]], \
													   [[8.,9.,10.], \
														[8.,9.,10.]]]])))
	assert(np.array_equal(is_terminal_batch, np.array([False, False, False, False, False])))

def main():
	module = __import__(inspect.getmodulename(__file__))
	for name in dir(module):
		attr = getattr(module, name)
		if inspect.isfunction(attr) and 'test' in attr.__name__:
			attr()
			print(attr.__name__ + ' passed')
			
if __name__ == '__main__':
    main()