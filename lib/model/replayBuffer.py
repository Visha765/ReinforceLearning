from collections import deque
import numpy as np

class ReplayBuffer():
  def __init__(self, buffer_size: int):
    self.que = []
    self.buffer_size = buffer_size
  
  def add(self, state, action, next_state, reward, done):
    exp = [state, action, next_state, reward, done]
    self.que.append(exp)
    if len(self.que) > self.buffer_size:
      self.que.pop(0)
      
  
  def sample(self, batch_size):
    size = min(len(self.que), batch_size)
    
    rand_nums = np.random.choice(len(self.que), size, replace=False)
    return [self.que[i] for i in rand_nums]

    