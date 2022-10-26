import numpy as np
import sys, os
import pickle
import random

from .agent import Agent
from .qTableAgent import QTableAgent
from .table import QTable
from .replayBuffer import ReplayBuffer

class ReplayQTableAgent(QTableAgent):
  
  def __init__(self, K, L, buffer_size, batch_size):
    
    super().__init__(K, L)
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    

  def train(self, state, action, next_state, reward, done):
    self.buffer.add(state, action, next_state, reward, done)
    for exp in self.buffer.sample(self.batch_size):
      state, action, next_state, reward, done = exp
      self.qTable.update_Qtable(state, action, reward, next_state, done)