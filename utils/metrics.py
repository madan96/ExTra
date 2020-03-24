import matplotlib.pyplot as plt
import numpy as np
import time

from env import Environment, create_env
from qlearn.agent import QLearningAgent

def match_actions(gt_agent, curr_agent, env):
    gt_policy_val = np.mean(gt_agent.qvalues)
    curr_policy_val = np.mean(curr_agent.qvalues)
    return np.fabs(gt_policy_val - curr_policy_val)

def evaluate_mean_avg_reward(env, agent):
    state = env.get_state()
    mean_avg_reward = 0
    num_episodes = 1000
    for i in range(env.state_space):
        env.start_position = env.idx2state[i]
        env.position = env.idx2state[i]
        avg_reward = 0
        # env.render(agent.qvalues)
        # time.sleep(0.1)
        for j in range(50):
            possible_actions= env.get_possible_actions()
            action = agent.get_best_action(state, possible_actions)
            next_state, reward, done, next_possible_states = env.step(action)

            next_state_possible_actions = env.get_possible_actions()
            state = next_state
            avg_reward += reward
            # env.render(agent.qvalues)
            if done == True:	
                break
        # time.sleep(0.1)
        avg_reward /= (j + 1)
        mean_avg_reward += avg_reward
        env.reset_state()
        state = env.get_state()
    return mean_avg_reward / env.state_space

def generate_heatmap(env, count_matrix, save_path):
    heatmap = np.zeros((env.gridH, env.gridW))
    # env.generate_heatmap(count_matrix, save_path)
    for i in range(env.state_space):
        x, y = env.idx2state[i]
        heatmap[x, y] = np.sum(count_matrix[i])
    f, ax = plt.subplots()
    im = ax.imshow(heatmap)
    # plt.show()
    plt.savefig(save_path)

def exploration_quality(optimal_q, count_matrix):
    prod = np.multiply(optimal_q, count_matrix)
    return np.sum(prod)