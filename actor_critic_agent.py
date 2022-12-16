import numpy as np
import sys, os
import pickle
import torch, torch.nn as nn

from agent import Agent
from replay_buffer import ReplayBuffer
from actor import ActorNet
from critic import CriticNet

class ActorCriticAgent(Agent):
  def __init__(self, K, L, buffer_size, batch_size, gamma=0.99, alpha=3*1e-4, epsilon=0.05):
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    self.gamma = gamma
    self.alpha = alpha
    self.epsilon = epsilon
    
    self.actor = ActorNet(K**2,L)
    self.critic = CriticNet(K**2, L)

  def save_models(self, env, current_step, path):
    data = {'agent': self, 'saved_step': current_step,}
    filename = f"log_{env.unwrapped.spec.id}_step{current_step}.pickle"
    with open(os.path.join(path, filename), 'wb') as f:
      pickle.dump(data, f)

  def load_models(self, path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
      data = pickle.load(f)
    return (data[key] for key in ('agent', 'saved_step'))

  def select_action(self, state):
    pass
    # return [self.qTable.get_maxQ_action(state)]
  
  def select_exploratory_action(self, state):
    pass
    # if self.epsilon < np.random.uniform(0,1):
    #   return self.select_action(state)
    # else:
    #   return np.random.choice(self.qTable.actions, 1)

  def train(self, state, action, next_state, reward, done):
    self.buffer.add(state, action, next_state, reward, done)
    delta = []
    for state, action, next_state, reward, done in self.buffer.sample(self.batch_size):
      delta.append(reward + (1-done) * self.gamma * self.critic.forward([next_state, self.select_action(next_state)]))
    criterion = nn.MSELoss()
    loss = criterion()

      