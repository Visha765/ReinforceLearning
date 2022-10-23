import gym
import numpy as np
import pandas as pd

# state: (cos(theta), sin(theta), omega) --> (theta, omega)
def xy2theta(state):
  return (np.arccos(state[0]), state[2])


def create_Qtable(K :int ,L :int):
  
  EPS = 1e-12
  
  # obs = env.observation_space
  bin_theta = np.linspace(0-EPS, 360, K+1) 
  bin_omega = np.linspace(-8-EPS, 8, K+1)
  bin_tau = np.linspace(-2-EPS, 2, L+1)
  
  theta_index = pd.IntervalIndex.from_breaks(bin_theta)
  omega_index = pd.IntervalIndex.from_breaks(bin_omega)
  tau_index = pd.IntervalIndex.from_breaks(bin_tau) 
  
  index_0 = pd.MultiIndex.from_product(
    [theta_index, omega_index],
    names=["theta_cat", "omega_cat"])
  # index_1 = pd.MultiIndex.from_product(
  #   [tau_index],
  #   names=["tau_cat"])
  index_1 = tau_index
  q_table = pd.DataFrame(1e-8*np.random.normal(size=(K*K, L)), index=index_0, columns=index_1)
  
  return q_table


def update_Qtable(q_table, state, action, reward, next_state):
  gamma = 0.99
  alpha = 3*1e-4
  
  state = xy2theta(state)
  next_state = xy2theta(next_state)
  next_Max_Q = q_table.loc[next_state, :].max()
  q_table.loc[state, action] = (1-alpha) * q_table.loc[state, action] + alpha * (reward + gamma * next_Max_Q)
  return q_table
  
  
# return action of Max.Q
def get_maxQ_action(q_table, state):
  state = xy2theta(state)
  return [q_table.loc[state, :].idxmax().mid]


#return actions
def get_actions(q_table, state):
  return [i.mid for i in q_table.columns]