import argparse
import datetime
import json
import numpy as np
import os
import pandas as pd
import random
import sys

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


from env import Environment, create_env
from utils import match_actions, evaluate_mean_avg_reward
from qlearn.agent import QLearningAgent

plot_freq = 10

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--mode',
        default='train',
        type=str
    )
    argparser.add_argument(
        '--env-name',
        default='FourLargeRooms',
        type=str
    )
    argparser.add_argument(
        '--alpha',
        default=0.2,
        type=float
    )
    argparser.add_argument(
        '--epsilon',
        default=0.1,
        type=float
    )
    argparser.add_argument(
        '--discount',
        default=0.99,
        type=float
    )
    argparser.add_argument(
        '--num-iters',
        default=10000,
        type=int
    )
    argparser.add_argument(
        '--policy-dir',
        default='saved_qvalues/optimal_qvalues',
        type=str
    )
    argparser.add_argument(
        '-ma', '--match-action',
        action='store_true',
        dest='debug',
        help='Match actions with ground truths and generate plots'
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    args = argparser.parse_args()

    env = create_env(args.env_name, 712)
    agent = QLearningAgent(args.alpha, args.epsilon, args.discount, env.action_space, env.state_space, env.tp_matrix, env.blocked_positions, 712)
    agent.qvalues = np.load(os.path.join(args.policy_dir, args.env_name + '.npy'))

    avg_reward_vec = []
    # for _ in range(args.num_iters):
    avg_r = evaluate_mean_avg_reward(env, agent)
    print (avg_r)
        # avg_reward_vec.append(avg_r)
    avg_reward_vec = np.asarray((avg_r)).repeat(int(args.num_iters / plot_freq))
    timesteps = np.arange(start=0, stop=args.num_iters, step=plot_freq)
    avg_reward_array = np.asarray(avg_reward_vec)
    print ("Aa", avg_reward_array[0])
    df = pd.DataFrame({'Timesteps':timesteps, 'Mean Average Reward':avg_reward_array})
    df_path = os.path.join('/home/rishabh/work/btp/TRAiL/tools/'+ args.env_name + '/' + args.env_name + '_optimal_mar_data.csv')
    df.to_csv(df_path)