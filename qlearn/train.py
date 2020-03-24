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
from utils import exploration_quality
from .agent import QLearningAgent

from scipy.interpolate import splrep, splev

plot_freq = 200
exp_check_thresh = 1000

def train(opts):
    log_path = 'output_logs/Q-Learning-Logs/{}'.format(opts.env_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    curr_log_dir = os.path.join(log_path, str(datetime.datetime.now())[:-7])
    os.makedirs(curr_log_dir)
    with open(os.path.join(curr_log_dir, 'command.txt'), 'w') as f:
        json.dump(opts.__dict__, f, indent=2)

    plt_path = os.path.join(curr_log_dir, opts.env_name + '_qlearn.png')
    policy_path = os.path.join(curr_log_dir, opts.env_name + '_qlearn.npy')
    df_path = os.path.join(curr_log_dir, opts.env_name + '_qlearn.csv')
    heatmap_path = os.path.join(curr_log_dir, opts.env_name + '_eps_explore.png')

    seeds = np.arange(opts.num_seeds)
    cr_allvec = []
    tp_all = []
    exp_q_all = []
    avg_reward_all = []
    pbar = tqdm(total=opts.num_seeds)
    dummy_env = create_env(opts.env_name, 712)
    count_matrix_avg = np.zeros((dummy_env.state_space, dummy_env.action_space))
    for i in seeds:
        np.random.seed(i)
        random.seed(i)

        env = create_env(opts.env_name, i)
        action_space = env.action_space
        state_space = env.state_space
        tp_matrix = env.tp_matrix
        count_matrix = np.zeros((env.state_space, env.action_space))

        gt_agent = QLearningAgent(opts.alpha, opts.epsilon, opts.discount, action_space, state_space, tp_matrix, env.blocked_positions, i)
        gt_agent.qvalues = np.load(os.path.join(opts.policy_dir, opts.env_name + '.npy'))
        gt_policy_val = np.mean(gt_agent.qvalues)
        agent = QLearningAgent(opts.alpha, opts.epsilon, opts.discount, action_space, state_space, tp_matrix, env.blocked_positions, i)
        agent.qvalues = np.random.rand(state_space, action_space)
        # env.render(agent.qvalues)

        cr_vec = []
        exp_q_vec = []
        c_reward = 0
        avg_reward_vec = []

        state = env.get_state()
        for i in range(opts.num_iters):
            possible_actions = env.get_possible_actions()
            action = agent.get_action(state, possible_actions)
            next_state, reward, done, next_possible_states = env.step(action)
            # env.render(agent.qvalues)
            if i < exp_check_thresh:
                count_matrix[state][action] = 1
            next_state_possible_actions = env.get_possible_actions()
            agent.update(state, action, reward, next_state, next_state_possible_actions, done)
            state = next_state

            c_reward += reward
            # pival_diff = match_actions(gt_agent, agent, env)
            if i % plot_freq == 0:
                avg_r = evaluate_mean_avg_reward(dummy_env, agent)
                exp_q = exploration_quality(gt_agent.qvalues, count_matrix)
                avg_reward_vec.append(avg_r)
                exp_q_vec.append(exp_q)

            if done == True:	
                env.reset_state()
                # env.render(agent.qvalues)
                state = env.get_state()
                continue

        avg_r = evaluate_mean_avg_reward(dummy_env, agent)
        # print ("Average reward: ", avg_r)
        # for _ in range(100):
        #     env.render(agent.qvalues)
        timesteps = np.arange(start=0, stop=opts.num_iters, step=plot_freq)
        tp_all.append(timesteps)
        avg_reward_all.append(avg_reward_vec)
        exp_q_all.append(exp_q_vec)
        pbar.update(1)
        count_matrix_avg += count_matrix
    
    
    temp_tp = np.arange(start=0, stop=opts.num_iters, step=plot_freq)
    count_matrix_avg /= opts.num_seeds
    timesteps = np.asarray(tp_all)
    avg_reward_array = np.asarray(avg_reward_all)
    exp_q_array = np.asarray(exp_q_all)
    timesteps = timesteps.reshape((opts.num_seeds * temp_tp.shape[0]))
    avg_reward_all = avg_reward_array.reshape((opts.num_seeds * temp_tp.shape[0]))
    exp_q_all = exp_q_array.reshape((opts.num_seeds * temp_tp.shape[0]))
    df = pd.DataFrame({'Timesteps':timesteps, 'Mean Average Reward':avg_reward_all, 'Exp Quality':exp_q_all})
    
    mean_avg_reward = np.mean(avg_reward_array, axis=0)
    exp_q = np.mean(exp_q_array, axis=0)
    tp = timesteps[:temp_tp.shape[0]]
    poly = np.polyfit(tp,mean_avg_reward,5)
    poly_y = np.poly1d(poly)(tp)

    plt.subplot(2, 1, 1)
    plt.title('Epsilon-Greedy')
    plt.grid()
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Average Reward')
    plt.plot(tp, poly_y)
    
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.xlabel('Timesteps')
    plt.ylabel('Exploration Quality')
    plt.plot(tp[:int(exp_check_thresh / plot_freq)], exp_q[:int(exp_check_thresh / plot_freq)])

    plt.savefig(plt_path)
    df.to_csv(df_path)
    np.save(policy_path, agent.qvalues)
    env.generate_heatmap(count_matrix, heatmap_path)