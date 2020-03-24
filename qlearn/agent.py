import numpy as np
import random
import scipy.special as sp

# ------------------------------------------------------------------------------------------
# ------------------------------------- Base Agent -----------------------------------------
# ------------------------------------------------------------------------------------------

class BaseAgent:
	
	def __init__(self, alpha, epsilon, discount, action_space, state_space, tp_matrix, blocked_positions, seed):
 
		self.action_space = action_space
		self.alpha = alpha
		self.epsilon = epsilon
		self.epsilon_bisim = 0.99
		self.discount = discount
		self.qvalues = np.zeros((state_space, action_space), np.float32)
		self.tp_matrix = tp_matrix
		self.seed = seed
		self.pi_softmax = np.zeros((state_space, action_space))
		self.pi_pursuit = np.ones((state_space, action_space)) / action_space
		np.random.seed(seed)
		
	def update(self, state, action, reward, next_state, next_state_possible_actions, done):

		# Q(s,a) = (1.0 - alpha) * Q(s,a) + alpha * (reward + discount * V(s'))

		if done==True:
			qval_dash = reward
		else:
			qval_dash = reward + self.discount * self.get_value(next_state, action, next_state_possible_actions)
			
		qval_old = self.qvalues[state][action]      
		qval = (1.0 - self.alpha)* qval_old + self.alpha * qval_dash
		self.qvalues[state][action] = qval
        
	def get_best_action(self, state, possible_actions):

		best_action = possible_actions[0]
		value = self.qvalues[state][possible_actions[0]]
        
		for action in possible_actions:
			q_val = self.qvalues[state][action]
			if q_val > value:
				value = q_val
				best_action = action

		return best_action

	def get_action(self, state, possible_actions):
         
		# with probability epsilon take random action, otherwise - the best policy action

		epsilon = self.epsilon

		if epsilon > np.random.uniform(0.0, 1.0):
			chosen_action = random.choice(possible_actions)
		else:
			chosen_action = self.get_best_action(state, possible_actions)

		return chosen_action
	
	def get_action_trexqlearn(self, state, possible_actions, lower_bound, d_sa_final, timestep, epsilon_bisim, temp):
         
		# with probability epsilon take random action, otherwise - the best policy action

		epsilon = self.epsilon
		alpha = 1e-6

		if epsilon > np.random.uniform(0.0, 1.0):
			s_t = np.argmax(lower_bound[:, state])
			# noise = np.random.random_sample(d_sa_final.shape)
			# d_sa_final += noise
			# print (d_sa_final[s_t, 0, state])
			prob = sp.softmax(-(d_sa_final[s_t, 0, state] + alpha * timestep))
			chosen_action = np.random.choice(np.arange(0, 4), p=prob)
		
			if epsilon_bisim > np.random.uniform(0.0, 1.0):
				return chosen_action
			else:
				chosen_action = random.choice(possible_actions)
		else:
			chosen_action = self.get_best_action(state, possible_actions)

		return chosen_action
	
	def get_action_pursuit(self, state, possible_actions, beta):
         
		# with probability epsilon take random action, otherwise - the best policy action
		chosen_action = self.get_best_action(state, possible_actions)
		self.pi_pursuit[state][chosen_action] = self.pi_pursuit[state][chosen_action] + beta * (1 - self.pi_pursuit[state][chosen_action])
		other_actions = [0, 1, 2, 3]
		other_actions.remove(chosen_action)

		self.pi_pursuit[state][other_actions] = self.pi_pursuit[state][other_actions] + beta * (0 - self.pi_pursuit[state][other_actions])

		action = np.random.choice(np.arange(0, 4), p=self.pi_pursuit[state] / np.sum(self.pi_pursuit[state]))
		return action
	
	# def get_action_softmax(self, state, possible_actions, lower_bound, d_sa_final, timestep, epsilon_bisim_matrix, temp):
	def get_action_softmax(self, state, possible_actions, temp):
		self.pi_softmax[state] = sp.softmax(self.qvalues[state])
		action = np.random.choice(np.arange(0, 4), p=self.pi_softmax[state] / np.sum(self.pi_softmax[state]))
		return action
	
	def get_action_trex(self, state, possible_actions, lower_bound, d_sa_final, timestep, epsilon_bisim_matrix, temp):
		s_t = np.argmax(lower_bound[:, state])
		# tau = 4000
		alpha = 1e-4 #max(1 - timestep / (2 * tau), 0.5) #(timestep - tau) / 10000.
		if timestep > 10000:
			alpha = 0.5
		# Normalize
		min_d = np.min(d_sa_final[s_t, 0, state])
		d = d_sa_final[s_t, 0, state] + alpha * timestep # (d_sa_final[s_t, 0, state] - min_d) / (np.max(d_sa_final[s_t, 0, state]) - min_d)
		
		min_q = np.min(self.qvalues[state])
		q = self.qvalues[state] # (self.qvalues[state] - min_q) / (np.max(self.qvalues[state]) - min_q)

		self.pi_softmax[state] = sp.softmax(q - d)
		action = np.random.choice(np.arange(0, 4), p=self.pi_softmax[state] / np.sum(self.pi_softmax[state]))
		# self.pi_softmax[state] = sp.softmax(self.qvalues[state])
		# action = np.random.choice(np.arange(0, 4), p=self.pi_softmax[state] / np.sum(self.pi_softmax[state]))
		return action
	

	def get_action_trail(self, state, possible_actions, lower_bound, d_sa_final, timestep, epsilon_bisim, temp):
		epsilon = self.epsilon
		alpha = 1e-6
		# print ('a')
		# if timestep > 10000:
		# 	epsilon = max(0, epsilon - (self.epsilon - 0.1) * (timestep - 10000) / 20000)
		if epsilon > np.random.uniform(0.0, 1.0):
			s_t = np.argmax(lower_bound[:, state])
			# noise = np.random.random_sample(d_sa_final.shape)
			# d_sa_final += noise
			# print (d_sa_final[s_t, 0, state])
			prob = sp.softmax(-(d_sa_final[s_t, 0, state] + alpha * timestep))
			chosen_action = np.random.choice(np.arange(0, 4), p=prob)
			# epsilon_bisim = decay_epsilon(self.epsilon_bisim, timestep)
			# b_t = np.argmin(d_sa_final[s_t, 0, state])
			# if epsilon_bisim > np.random.uniform(0.0, 1.0):
			# 	chosen_action = b_t
			# else:
			# 	chosen_action = random.choice(possible_actions)
		else:
			# self.pi_softmax[state] = sp.softmax(self.qvalues[state])
			# chosen_action = np.random.choice(np.arange(0, 4), p=self.pi_softmax[state] / np.sum(self.pi_softmax[state]))
			chosen_action = self.get_best_action(state, possible_actions)
		return chosen_action
	
	def get_action_trail_2(self, state, possible_actions, lower_bound, d_sa_final, timestep, epsilon_bisim_matrix):
		s_t = np.argmax(lower_bound[:, state])
		prob = sp.softmax(1/d_sa_final[s_t, 0, state])
		b_t = np.random.choice(np.arange(0, 4), p=prob)

		epsilon = self.epsilon
		if epsilon_bisim_matrix[b_t] > np.random.uniform(0.0, 1.0):
			chosen_action = b_t
		# epsilon = self.epsilon - (0.0/99 * timestep) / 10000
		# if timestep > 5000:
		# 	epsilon = 0.05
		# if epsilon > np.random.uniform(0.0, 1.0):
		# 	s_t = np.argmax(lower_bound[:, state])
		# 	# noise = np.random.random_sample(d_sa_final.shape)
		# 	# d_sa_final += noise
		# 	b_t = np.argmin(d_sa_final[s_t, 0, state])
			
		# 	# epsilon_bisim = decay_epsilon(self.epsilon_bisim, timestep)
			
		# 	if epsilon_bisim_matrix[b_t] > np.random.uniform(0.0, 1.0):
		# 		chosen_action = b_t
		# 	else:
		# 		chosen_action = random.choice(possible_actions)
		else:
			# self.pi_softmax[state] = sp.softmax(self.qvalues[state] / 1)
			# chosen_action = np.random.choice(np.arange(0, 4), p=self.pi_softmax[state] / np.sum(self.pi_softmax[state]))
			chosen_action = self.get_best_action(state, possible_actions)
		
	def get_action_trexpursuit(self, state, possible_actions, lower_bound, d_sa_final, timestep, temp, beta):
		epsilon = self.epsilon
		alpha = 1e-8
		# print ('a')
		# if timestep > 10000:
		# 	epsilon = max(0, epsilon - (self.epsilon - 0.1) * (timestep - 10000) / 20000)
		if epsilon > np.random.uniform(0.0, 1.0):
			s_t = np.argmax(lower_bound[:, state])
			# noise = np.random.random_sample(d_sa_final.shape)
			# d_sa_final += noise
			# print (d_sa_final[s_t, 0, state])
			prob = sp.softmax(-(d_sa_final[s_t, 0, state] + alpha * timestep))
			chosen_action = np.random.choice(np.arange(0, 4), p=prob)
			# epsilon_bisim = decay_epsilon(self.epsilon_bisim, timestep)
			# b_t = np.argmin(d_sa_final[s_t, 0, state])
			# if epsilon_bisim > np.random.uniform(0.0, 1.0):
			# 	chosen_action = b_t
			# else:
			# 	chosen_action = random.choice(possible_actions)
		else:
			chosen_action = self.get_best_action(state, possible_actions)
			self.pi_pursuit[state][chosen_action] = self.pi_pursuit[state][chosen_action] + beta * (1 - self.pi_pursuit[state][chosen_action])
			other_actions = [0, 1, 2, 3]
			other_actions.remove(chosen_action)

			self.pi_pursuit[state][other_actions] = self.pi_pursuit[state][other_actions] + beta * (0 - self.pi_pursuit[state][other_actions])

			chosen_action = np.random.choice(np.arange(0, 4), p=self.pi_pursuit[state] / np.sum(self.pi_pursuit[state]))

		return chosen_action
	
	def get_action_trexsoftmax(self, state, possible_actions, lower_bound, d_sa_final, timestep, temp):
		epsilon = self.epsilon
		alpha = 1e-6
		# print ('a')
		# if timestep > 10000:
		# 	epsilon = max(0, epsilon - (self.epsilon - 0.1) * (timestep - 10000) / 20000)
		if epsilon > np.random.uniform(0.0, 1.0):
			s_t = np.argmax(lower_bound[:, state])
			# noise = np.random.random_sample(d_sa_final.shape)
			# d_sa_final += noise
			# print (d_sa_final[s_t, 0, state])
			prob = sp.softmax(-(d_sa_final[s_t, 0, state] + alpha * timestep))
			chosen_action = np.random.choice(np.arange(0, 4), p=prob)
			# epsilon_bisim = decay_epsilon(self.epsilon_bisim, timestep)
			# b_t = np.argmin(d_sa_final[s_t, 0, state])
			# if epsilon_bisim > np.random.uniform(0.0, 1.0):
			# 	chosen_action = b_t
			# else:
			# 	chosen_action = random.choice(possible_actions)
		else:
			self.pi_softmax[state] = sp.softmax(self.qvalues[state] / temp)
			chosen_action = np.random.choice(np.arange(0, 4), p=self.pi_softmax[state] / np.sum(self.pi_softmax[state]))
		return chosen_action

	def update_qvalue(self, state, action, value):
		self.qvalues[state][action] = value
		return
   
	def get_value(self, state, next_state, action, next_state_possible_actions, next_possible_states):		
		pass

# ------------------------------------------------------------------------------------------
# ---------------------------------- Q-Learning Agent --------------------------------------
# ------------------------------------------------------------------------------------------

class QLearningAgent(BaseAgent):

	def get_value(self, state, action, possible_actions):

		# estimate V(s) as maximum of Q(state,action) over possible actions

		value = self.qvalues[state][possible_actions[0]]
		
		# value_sum = 0.
		# ps = np.nonzero(self.tp_matrix[state][action])[0]
		# 	# print ("ps", possible_states)
		# for s_dash in ps:
		# 	a_dash = np.argmax(self.qvalues[s_dash])
		# 	value_sum += self.qvalues[s_dash][a_dash] * self.tp_matrix[state][action][s_dash]

		for action in possible_actions:
			q_val = self.qvalues[state][action]
			if q_val > value:
				value = q_val

		return value
	# def get_value(self, state, next_state, action_taken, possible_actions, next_possible_states):

	# 	# estimate V(s) as maximum of Q(state,action) over possible actions
	# 	value_sum = 0.
	# 	for s in next_possible_states:
	# 		value = self.qvalues[s][possible_actions[0]]
		
	# 		for action in possible_actions:
	# 			q_val = self.qvalues[s][action]
	# 			if q_val > value:
	# 				value = q_val
			
	# 		value = self.tp_matrix[state][action_taken][s] * value
	# 		value_sum += value

	# 	return np.max(self.qvalues[state])

# ------------------------------------------------------------------------------------------
# ------------------------------ Expected Value SARSA Agent --------------------------------
# ------------------------------------------------------------------------------------------
    
class EVSarsaAgent(BaseAgent):
    
	def get_value(self, state, possible_actions):
		
		# estimate V(s) as expected value of Q(state,action) over possible actions assuming epsilon-greedy policy
		# V(s) = sum [ p(a|s) * Q(s,a) ]
          
		best_action = possible_actions[0]
		max_val = self.qvalues[state][possible_actions[0]]
		
		for action in possible_actions:
            
			q_val = self.qvalues[state][action]
			if q_val > max_val:
				max_val = q_val
				best_action = action
        
		state_value = 0.0
		n_actions = len(possible_actions)
		
		for action in possible_actions:
            
			if action == best_action:
				trans_prob = 1.0 - self.epsilon + self.epsilon/n_actions
			else:
				trans_prob = self.epsilon/n_actions
                   
			state_value = state_value + trans_prob * self.qvalues[state][action]

		return state_value

def decay_epsilon(epsilon, timestep):
	if timestep > 1000:
		return epsilon
	epsilon = epsilon - 0.0 * (timestep / 1000)
	return epsilon