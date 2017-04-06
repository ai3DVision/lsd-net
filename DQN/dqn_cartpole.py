#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
import gym

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam

import deeprl_hw2 as tfrl
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import huber_loss
from deeprl_hw2.preprocessors import AtariPreprocessor
from deeprl_hw2.policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.memory import BasicMemory, NaiveMemory
from deeprl_hw2.constants import model_path, model_file
from deeprl_hw2.models import create_model
from deeprl_hw2.utils import get_output_folder
from deeprl_hw2.core import Preprocessor

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on CartPole')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--phase', default='train', type=str, help='Train/Test/Video')
    parser.add_argument('-r', '--render', action='store_true', default=False, help='Render')
    parser.add_argument('--model', default='cartpole', type=str, help='Type of model')
    parser.add_argument('-c', action='store_false', default=True, help='Cancel')
    parser.add_argument('-d', '--dir', default='', type=str, help='Directory')
    parser.add_argument('-n', '--number', default='', type=str, help='Model number')
    parser.add_argument('--double', action='store_true', default=False, help='Cancel')

    args = parser.parse_args()

    assert(args.phase in ['train', 'test', 'video'])
    assert(args.dir if args.phase == 'test' or args.phase == 'video' else True)

    env = 'CartPole-v0'
    output_dir = get_output_folder(args.output, env) \
                if not args.dir \
                else os.path.join(args.output, args.dir)
    args.model = 'double_' + args.model if args.double else args.model
    
    # create the environment
    env = gym.make(env)
    args.input_shape = env.observation_space.shape

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    # Number of training iterations
    num_iterations = 25000

    # Learning rate
    alpha = 0.001

    # Epsilion for GreedyEpsilonPolicy
    epsilon = 0.05

    # Parameters for LinearDecayGreedyEpsilonPolicy
    start_value = 1
    end_value = 0.1
    num_steps = 10000

    # Number of frames in the sequence
    window = 1

    # Use experience replay
    experience_replay = args.c

    # Use target fixing
    target_fixing = args.c

    # Evaluate number of episode (given the model number)
    num_episode = 100

    # DQNAgent parameters
    num_actions = env.action_space.n
    q_network = create_model(window, 
                             args.input_shape, 
                             num_actions, 
                             model_name=args.model)
    preprocessor = Preprocessor()
    policy = LinearDecayGreedyEpsilonPolicy(num_actions, start_value, end_value, num_steps)
    memory_size = 10000
    gamma = 0.99
    target_update_freq = 100
    num_burn_in = 50
    train_freq = 4
    batch_size = 32
    save_network_freq = 100
    video_capture_points = (num_iterations * np.array([0/3., 1/3., 2/3., 3/3.])).astype('int')
    eval_train_freq = 2500
    eval_train_num_ep = 5
    print_summary = True

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
                        render=args.render,
                        print_summary=print_summary,
                        max_grad=np.inf,
                        double_dqn=args.double)
    dqnAgent.compile(Adam(lr=alpha), huber_loss, output=output_dir)

    if args.dir:
        model = model_file % (args.model, args.number)
        model_dir = os.path.join(args.output, args.dir, model_path, model)
        dqnAgent.q_network.load_weights(model_dir)
        
    if args.phase == 'train':
        dqnAgent.fit(env, num_iterations)
    elif args.phase == 'test':
        dqnAgent.policy = GreedyEpsilonPolicy(epsilon, num_actions)
        dqnAgent.evaluate(env, num_episode)
    elif args.phase == 'video':
        dqnAgent.policy = GreedyEpsilonPolicy(epsilon, num_actions)
        points = [''] + [point for point in video_capture_points]
        for point in points:
            model = model_file % (args.model, point)
            model_dir = os.path.join(args.output, args.dir, model_path, model)
            dqnAgent.q_network.load_weights(model_dir)
            dqnAgent.capture_episode_video(env, video_name=model)
        
if __name__ == '__main__':
    main()