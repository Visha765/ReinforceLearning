import gym
import numpy as np
import pandas as pd

class QTable():
  EPS = 1e-12
  
  
  def __init__(self, K :int ,L :int):
    self.gamma = 0.99
    self.alpha = 3*1e-4
    
    # obs = env.observation_space
    bin_theta = np.linspace(0-self.EPS, 360, K+1) 
    bin_omega = np.linspace(-8-self.EPS, 8, K+1)
    bin_tau = np.linspace(-2-self.EPS, 2, L+1)
    
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
    self.table = pd.DataFrame(1e-8*np.random.normal(size=(K*K, L)), index=index_0, columns=index_1)


  def update_Qtable(self, state, action, reward, next_state, done=0):
    state = self.xy2theta(state)
    next_state = self.xy2theta(next_state)
    next_Max_Q = self.table.loc[next_state, :].max()
    self.table.loc[state, action] = (1-self.alpha) * self.table.loc[state, action] \
      + self.alpha * (reward + (1 - done) * self.gamma * next_Max_Q)
    
    
  # return action of Max.Q
  def get_maxQ_action(self, state):
    state = self.xy2theta(state)
    return [self.table.loc[state, :].idxmax().mid]


  #return actions
  def get_actions(self, state):
    return [i.mid for i in self.table.columns]
  
  
    # state: (cos(theta), sin(theta), omega) --> (theta, omega)
  def xy2theta(self, state):
    return (np.arccos(state[0]), state[2])