import numpy as np
import sys, os
import pickle
import torch, torch.nn as nn
import copy

from lib.model.agent import Agent
from lib.model.replay_buffer import ReplayBuffer
from lib.model.actor import ActorNet
from lib.model.critic import CriticNet
from lib.util.xy2theta import xy2theta

class ActorCriticAgent(Agent):
  def __init__(self, buffer_size, batch_size, gamma=0.99, sigma_beta=0.1, T_expl=1000):
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    self.gamma = gamma
    self.sigma_beta = sigma_beta
    self.T_expl = T_expl
    
    self.actor = ActorNet(2, 1)
    self.critic = CriticNet(2, 1)
    
    self.criterion_critic = nn.MSELoss()
    self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=0.01)
    self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=0.01)
    

  def save_models(self, current_step, path):
    filename = "log_step{:07d}.pickle".format(current_step)
    tmp = copy.deepcopy(self.buffer)
    self.buffer = []
    with open(os.path.join(path, filename), 'wb') as f:
      pickle.dump(self, f)
    self.buffer = tmp

  def load_models(self, path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
      data = pickle.load(f)
    return data

  def select_action(self, state):
    state = xy2theta(state)
    return self.actor(self.list2tensor([state,])).detach().numpy()[0] #tensor -> ndarray
  
  def select_exploratory_action(self, state, current_step):
    tau = (-2, 2)
    if current_step < self.T_expl:
      return np.random.uniform(*tau, 1)
    
    D = (tau[1] - tau[0]) * self.sigma_beta/2
    return self.select_action(state) + np.random.normal(0, D**2)

  def train(self, state, action, next_state, reward, done):
    state = xy2theta(state)
    next_state = xy2theta(next_state)
    self.buffer.add(state, action, next_state, reward, done)
    
    states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)

    # critic    
    next_states = self.list2tensor(next_states)
    next_actions = self.actor(next_states)

    Q_actor = self.critic(torch.cat([next_states, next_actions], dim=1))
    delta = self.list2tensor(rewards).view(-1,1) \
      + self.gamma * torch.mul(self.list2tensor(np.logical_not(dones)).view(-1,1), Q_actor)

    states = self.list2tensor(states)
    actions = self.list2tensor(actions)
    Q = self.critic(torch.cat([states, actions], dim=1))
    loss_omega = self.criterion_critic(Q, delta)
    loss_omega.backward()

    self.optimizer_critic.zero_grad()
    self.optimizer_critic.step()
    
    # actor    
    actions = self.actor(states)
    
    loss_theta = torch.mean(-self.critic(torch.cat([states, actions], dim=1)))
    loss_theta.backward()
    self.optimizer_actor.zero_grad()
    self.optimizer_actor.step()
    
    
  def list2tensor(self, x):
    x = np.array(x)
    return torch.Tensor(x)
    
