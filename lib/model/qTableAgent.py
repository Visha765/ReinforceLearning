import numpy as np
import sys, os
import pickle

from .agent import Agent
from .qTable import QTable

class QTableAgent(Agent):
  def __init__(self, K, L):
    self.epsilon = 0.05
    self.qTable = QTable(K, L)
        
    
  def save_models(self, env, current_step, seed, path):
    print('saved step:', current_step)
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
      data = pickle.load(f)
    return (data[key] for key in ('env', 'agent', 'saved_step'))

  # PI
  def select_action(self, state):
    return [self.qTable.get_maxQ_action(state)]
  
  # beta
  def select_exploratory_action(self, state):
    if self.epsilon < np.random.uniform(0,1):
      return self.select_action(state)
    else:
      return np.random.choice(self.qTable.get_actions(state), 1)
        

  def train(self, state, action, next_state, reward, done):
    self.qTable.update_Qtable(state, action, reward, next_state)