# import os
# import pickle
import copy
from .qTableAgent import QTableAgent
from .replayBuffer import ReplayBuffer

class ReplayQTableAgent(QTableAgent):
  
  def __init__(self, K, L, buffer_size, batch_size, gamma=0.99, alpha=3*1e-4, epsilon=0.05):
    super().__init__(K, L, gamma, alpha, epsilon)
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    
  def save_models(self, env, current_step, seed, path):
    tmp = copy.deepcopy(self.buffer)
    self.buffer = []
    super().save_models(env, current_step, seed, path)
    self.buffer = tmp

  def train(self, state, action, next_state, reward, done):
    self.buffer.add(state, action, next_state, reward, done)
    for exp in self.buffer.sample(self.batch_size):
      state, action, next_state, reward, done = exp
      self.qTable.update_qTable(state, action, reward, next_state, done)