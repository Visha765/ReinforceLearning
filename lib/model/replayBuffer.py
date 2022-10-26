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
    
    exp_list = random.sample(self.que, size)
    return exp_list
    
    # que0 = copy.deepcopy(self.que)
    # shuffle_que = random.shuffle(que0)
    # print(type(self.que), self.que)
    # print(type(que0), que0)
    
    # shuffle_que = random.sample(self.que, )
    # return [que0[i] for i in range(size)]