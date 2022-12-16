from collections import deque
import numpy as np

class ReplayBuffer():
  def __init__(self, buffer_size: int):
    self.buffer_size = buffer_size
    self.que = []
  
  def add(self, state, action, next_state, reward, done):
    exp = [state, action, next_state, reward, done]
    self.que.append(exp)
    if len(self.que) > self.buffer_size:
      self.que.pop(0)
      
  def sample(self, batch_size):
    size = min(len(self.que), batch_size)
    rand_nums = np.random.choice(len(self.que), size, replace=False)
    # return [self.que[i] for i in rand_nums]
    dic = {"state":[], "action":[], "next_state":[], "reward":[], "done":[]}
    for i in rand_nums:
      exp = self.que[i]
      dic["state"].append(exp[0])
      dic["action"].append(exp[1])
      dic["next_state"].append(exp[2])
      dic["reward"].append(exp[3])
      dic["done"].append(exp[4])