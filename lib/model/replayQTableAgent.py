import os
import pickle
import copy
from .qTableAgent import QTableAgent
from .replayBuffer import ReplayBuffer

class ReplayQTableAgent(QTableAgent):
  
  def __init__(self, K, L, buffer_size, batch_size):
    
    super().__init__(K, L)
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    
  def save_models(self, env, current_step, seed, path):
    print('saved step:', current_step)
    data = {
            'agent': copy.deepcopy(self), 
            'saved_step': current_step,
          }
    data['agent'].buffer = []
    filename = f"log_{env.unwrapped.spec.id}_seed{seed}_step{current_step}.pikle"
    with open(os.path.join(path, filename), 'wb') as f:
      pickle.dump(data, f)
    

  def train(self, state, action, next_state, reward, done):
    self.buffer.add(state, action, next_state, reward, done)
    for exp in self.buffer.sample(self.batch_size):
      state, action, next_state, reward, done = exp
      self.qTable.update_Qtable(state, action, reward, next_state, done)