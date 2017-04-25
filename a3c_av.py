import tensorflow as tf
import keras.backend as K
from A3C.a3c.a3c_agent import A3CAgent
import gym
from A3C.a3c.env.nbv_env import NBVEnvironment
from A3C.a3c.utils import get_output_folder, mean_huber_loss
import nbv.envs

flags = tf.app.flags
flags.DEFINE_string('output', 'nbv', 'Name of the output folder')
flags.DEFINE_string('env', 'Next-Best-View-v0', 'Name of the environment')
flags.DEFINE_string('model', 'nbv', 'type of network')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_integer('checkpoint_interval', 5000, 'Checkpoint the model every n iterations')
flags.DEFINE_integer('summary_interval', 5, 'Save training summary to file every n iterations')
flags.DEFINE_boolean('show_training', False, 'If true, then render evironments during training')
flags.DEFINE_integer('num_concurrent', 32, 'Number of concurrent threads during training.')
flags.DEFINE_integer('agent_history_length', 3, 'History window length.')
flags.DEFINE_integer('resized_width', 224, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 224, 'Scale screen to this height.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_integer('num_iterations', 25000000, 'Number of training timesteps.')
flags.DEFINE_integer('async_update', 32, 'Size of batch to update network')
FLAGS = flags.FLAGS

def main(_):
	output_dir = get_output_folder(FLAGS.output, FLAGS.env)

	if FLAGS.testing:
		FLAGS.num_concurrent = 1
		
	# Set up game environments (one per thread)
	envs = [gym.make(FLAGS.env) for _ in range(FLAGS.num_concurrent)]
	envs = [NBVEnvironment(gym_env=env, \
						   resized_width=FLAGS.resized_width, \
						   resized_height=FLAGS.resized_height, \
						   agent_history_length=FLAGS.agent_history_length)
						   for env in envs]
	num_actions = envs[0].env.action_space.n
	input_shape = (FLAGS.resized_width, FLAGS.resized_height)

	with tf.Graph().as_default(), tf.Session() as session:
		K.set_session(session)

		a3cAgent = A3CAgent(model_name=FLAGS.model, 
							checkpoint_interval=FLAGS.checkpoint_interval, 
							summary_interval=FLAGS.summary_interval, 
							show_training=FLAGS.show_training, 
							num_concurrent=FLAGS.num_concurrent, 
							agent_history_length=FLAGS.agent_history_length, 
							input_shape=input_shape, 
							gamma=FLAGS.gamma, 
							learning_rate=FLAGS.learning_rate, 
							num_iterations=FLAGS.num_iterations, 
							async_update=FLAGS.async_update, 
							num_actions=num_actions, 
							output_dir=output_dir, 
							max_grad=1.)
		a3cAgent.compile(mean_huber_loss)
		saver = tf.train.Saver()

		if FLAGS.testing:
			a3cAgent.evaluation(envs[0], session, saver)
		else:
			a3cAgent.train(envs, session, saver)

if __name__ == "__main__":
	tf.app.run()