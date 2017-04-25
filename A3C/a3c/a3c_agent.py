#!/usr/bin/env python
from skimage.transform import resize
from skimage.color import rgb2gray
import threading
import tensorflow as tf
import sys
import random
import numpy as np
import time
import gym
from keras import backend as K
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque
import sys
if sys.version_info[0] == 2:
    from model.model import build_policy_and_value_networks
else:
    from A3C.a3c.model.model import build_policy_and_value_networks
from keras import backend as K
import os

class A3CAgent:
    summary_save_path = './%s/logs'
    model_save_path = './%s/models'
    checkpoint_name = '%s.ckpt'
    video_save_path = './%s/video'

    iteration = 0
    
    def __init__(self, 
                 model_name, 
                 checkpoint_interval, 
                 summary_interval, 
                 show_training, 
                 num_concurrent, 
                 agent_history_length, 
                 input_shape,  
                 gamma, 
                 learning_rate, 
                 num_iterations, 
                 async_update,
                 num_actions, 
                 output_dir,
                 max_grad):
        self.model_name = model_name
        self.checkpoint_interval = checkpoint_interval
        self.summary_interval = summary_interval
        self.show_training = show_training
        self.num_concurrent = num_concurrent
        self.agent_history_length = agent_history_length
        self.input_shape = input_shape
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.async_update = async_update
        self.num_actions = num_actions
        self.output_dir = output_dir
        self.max_grad = max_grad

        self.summary_save_path = self.summary_save_path % self.output_dir
        self.model_save_path = self.model_save_path % self.output_dir
        self.checkpoint_name = self.checkpoint_name % self.model_name
        self.video_save_path = self.video_save_path % self.output_dir

        self.checkpoint_save_path = os.path.join(self.model_save_path, self.checkpoint_name)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.video_save_path):
            os.makedirs(self.video_save_path)

    def sample_policy_action(self, num_actions, probs):
        probs = probs - np.finfo(np.float32).epsneg
        histogram = np.random.multinomial(1, probs)
        action = int(np.nonzero(histogram)[0])
        return action

    def actor_learner_thread(self, num, env, session, model, summary_ops, saver):
        state_input, action_input, target_input, minimize, p_network, v_network = model

        r_summary_placeholder, \
        update_ep_reward, \
        val_summary_placeholder, \
        update_ep_val, \
        pol_summary_placeholder, \
        update_ep_pol, \
        summary_op = summary_ops

        discount = 1
        discounted_ep_reward = 0
        ep_reward = 0
        ep_avg_v = 0
        ep_max_p = 0
        v_steps = 0
        ep_iters = 0

        state = env.reset()
        terminal = False

        while self.iteration < self.num_iterations:
            state_batch = []
            past_rewards = []
            action_batch = []

            async_update_count = 0

            while not (terminal or async_update_count  == self.async_update):
                probs = session.run(p_network, feed_dict={state_input: [state]})[0]
                action = self.sample_policy_action(self.num_actions, probs)
                action_mask = np.zeros([self.num_actions])
                action_mask[action] = 1

                state_batch.append(state)
                action_batch.append(action_mask)

                state, reward, terminal, info = env.step(action)
                ep_reward += reward

                discounted_ep_reward += reward * discount
                discount *= self.gamma

                reward = np.clip(reward, -1, 1)
                past_rewards.append(reward)

                max_p = np.max(probs)
                ep_max_p = ep_max_p + max_p

                async_update_count += 1
                self.iteration += 1
                ep_iters += 1

            if terminal:
                target = 0
            else:
                target = session.run(v_network, feed_dict={state_input: [state]})[0][0]

            ep_avg_v = ep_avg_v + target
            v_steps = v_steps + 1

            target_batch = np.zeros(async_update_count)
            for i in reversed(range(async_update_count)):
                target_batch[i] = past_rewards[i] + self.gamma * target

            session.run(minimize, feed_dict={target_input: target_batch,
                                             action_input: action_batch,
                                             state_input: state_batch})
            
            if self.iteration % self.checkpoint_interval == 0:
                saver.save(session, self.checkpoint_save_path)

            if terminal:
                if v_steps > 0:
                    session.run(update_ep_val, feed_dict={val_summary_placeholder: ep_avg_v/v_steps})
                if ep_iters > 0:
                    session.run(update_ep_pol, feed_dict={pol_summary_placeholder: ep_max_p/ep_iters})
                session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
                print('THREAD:', num, '/ TIME', self.iteration, '/ REWARD', ep_reward, '/ DISCOUNTED REWARD', discounted_ep_reward)
                state = env.reset()
                terminal = False
                discount = 1
                discounted_ep_reward = 0
                ep_reward = 0
                ep_iters = 0
                ep_avg_v = 0
                v_steps = 0
                ep_max_p = 0

    def compile(self, loss_func):
        state, \
        p_network, \
        v_network, \
        p_params, \
        v_params = build_policy_and_value_networks(model_name=self.model_name, \
                                                   num_actions=self.num_actions, \
                                                   input_shape=self.input_shape, \
                                                   window=self.agent_history_length)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        target = tf.placeholder('float', [None])
        action_mask = tf.placeholder('float', [None, self.num_actions])

        v_network_flat = tf.reshape(v_network, shape=[-1]);
        p_network_masked = tf.reduce_sum(tf.multiply(p_network, action_mask), reduction_indices=1)
        p_loss = loss_func(p_network_masked, target - v_network_flat, max_grad=self.max_grad)
        v_loss = tf.reduce_mean(tf.square(target - v_network)) / 2

        total_loss = p_loss + v_loss
        minimize = optimizer.minimize(total_loss)
        
        self.model = state, action_mask, target, minimize, p_network, v_network

    def setup_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar('Episode Reward', episode_reward)
        r_summary_placeholder = tf.placeholder('float')
        update_ep_reward = episode_reward.assign(r_summary_placeholder)
        
        ep_avg_v = tf.Variable(0.)
        tf.summary.scalar('Episode Value', ep_avg_v)
        val_summary_placeholder = tf.placeholder('float')
        update_ep_val = ep_avg_v.assign(val_summary_placeholder)

        ep_max_p = tf.Variable(0.)
        tf.summary.scalar('Episode Max Policy', ep_max_p)
        pol_summary_placeholder = tf.placeholder('float')
        update_ep_pol = ep_max_p.assign(pol_summary_placeholder)

        summary_op = tf.summary.merge_all()
        return r_summary_placeholder, \
               update_ep_reward, \
               val_summary_placeholder, \
               update_ep_val, \
               pol_summary_placeholder, \
               update_ep_pol, \
               summary_op

    def train(self, envs, session, saver):        
        summary_ops = self.setup_summaries()
        summary_op = summary_ops[-1]

        session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.summary_save_path, session.graph)

        actor_learner_threads = [threading.Thread(target=self.actor_learner_thread, \
                                                  args=(thread_id, \
                                                        envs[thread_id], \
                                                        session, \
                                                        self.model, \
                                                        summary_ops, \
                                                        saver)) for thread_id in range(self.num_concurrent)]
        for thread in actor_learner_threads:
            thread.start()

        last_summary_time = 0
        while True:
            if self.show_training:
                for env in envs:
                    env.render()
            now = time.time()
            if now - last_summary_time > self.summary_interval:
                summary_str = session.run(summary_op)
                writer.add_summary(summary_str, float(self.iteration))
                last_summary_time = now
        for thread in actor_learner_threads:
            thread.join()

    def evaluation(self, monitor_env, session, saver):
        saver.restore(session, self.checkpoint_save_path)
        print('Restored model weights from ', self.checkpoint_save_path)
        monitor_env.monitor.start(self.video_save_path)

        state_input, action_mask, target, minimize, p_network, v_network = self.model

        for i_episode in range(100):
            state = env.reset()
            ep_reward = 0
            terminal = False
            while not terminal:
                monitor_env.render()
                probs = p_network.eval(session = session, feed_dict = {state_input: [state]})[0]
                action = self.sample_policy_action(self.num_actions, probs)
                state, reward, terminal, info = env.step(action)
                ep_reward += reward
            print(ep_reward)
        monitor_env.monitor.close()
