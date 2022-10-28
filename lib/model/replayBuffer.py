from collections import deque
import numpy as np
import random
import copy

class ReplayBuffer():
  def __init__(self, buffer_size):
    self.que = deque([], buffer_size)
  
  def add(self, state, action, next_state, reward, done):
    exp = [state, action, next_state, reward, done]
    self.que.append(exp)
  
  def sample(self, batch_size):
    # if len(self.que) >= batch_size:
    size = min(len(self.que), batch_size)
    
    # exp_list = random.sample(self.que, size)
    rand_nums = np.random.choice(len(self.que), size, replace=False)
    return [self.que[i] for i in rand_nums]
    
    # exp_list = np.random.choice(self.que, size, replace=False)
    # return exp_list
    