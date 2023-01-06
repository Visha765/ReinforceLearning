from collections import deque
import numpy as np

items = ('state', 'action', 'next_state', 'reward', 'done')

class ReplayBuffer():
  def __init__(self, buffer_size: int):
    self.buffer_size = buffer_size
    self.que = deque([], maxlen=buffer_size)
    
  def __len__(self):
    return len(self.que)
  
  def add(self, state, action, next_state, reward, done):
    exp = {}
    args = (state, action, next_state, reward, done)
    for item, val in zip(items, args):
      exp[item] = val
    self.que.append(exp)
      
  def sample(self, batch_size):
    rand_nums = np.random.choice(len(self.que), batch_size, replace=False)
    out = []
    for item in items:
      out.append([self.que[i][item] for i in rand_nums])

    return out