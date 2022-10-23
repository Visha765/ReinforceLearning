import sys, os
import pickle
import numpy as np
from ..model.agent import Agent



class RandomAgent(Agent):
  
  def __init__(self, env):
    self.env = env;
    
  def save_models(self, env, current_step, seed, path):
    print('saved:', current_step)
    data = {'env': env, 
            'agent': self, 
            'saved_step': current_step,
            'seed': seed,
            # 'info': info
          }
    filename = f"log_{env.unwrapped.spec.id}_seed{seed}_step{current_step}.pikle"
    with open(os.path.join(path, filename), 'wb') as f:
      pickle.dump(data, f)
    
    
  def load_models(self, path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
    # with open(path, 'rb') as f:
      data = pickle.load(f)
    return (data[key] for key in ('env', 'agent', 'saved_step'))


  # def add_reward(self, reward):
  #   self.sum_reward += reward
  
  # PI
  def select_action(self, state):
    # print(np.shape(np.median([self.env.action_space.low, self.env.action_space.high], axis=0)))
    return np.median([self.env.action_space.low, self.env.action_space.high], axis=0)
  
  # beta
  def select_exploratory_action(self, state):
    return self.env.action_space.sample()
    # return np.random.rand() * (self._action_space[1] - self._action_space[0]) + self._action_space[0]

  def train(self, state, action, next_state, reward, done):
    pass