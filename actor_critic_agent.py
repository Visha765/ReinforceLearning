import numpy as np
import sys, os
import pickle
import torch, torch.nn as nn

from agent import Agent
from replay_buffer import ReplayBuffer
from actor import ActorNet
from critic import CriticNet
from xy2theta import xy2theta

class ActorCriticAgent(Agent):
  def __init__(self, buffer_size, batch_size, gamma=0.99, sigma_beta=0.1):
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    self.gamma = gamma
    self.sigma_beta = sigma_beta
    
    self.actor = ActorNet(2, 1)
    self.critic = CriticNet(2, 1)

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
    state = xy2theta(state)
    return self.actor(torch.tensor([state,])).detach().numpy()[0]
  
  def select_exploratory_action(self, state, current_step):
    T_expl = 1000
    tau = (-2, 2)
    
    if current_step < T_expl:
      return np.random.uniform(*tau, 1)
    
    D = (tau[1] - tau[0]) * self.sigma_beta/2
    return self.select_action(state) + np.random.normal(0, D**2)

  def train(self, state, action, next_state, reward, done):
    state = xy2theta(state)
    next_state = xy2theta(next_state)
    self.buffer.add(state, action, next_state, reward, done)
    
    states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)

    # critic
    criterion_omega = nn.MSELoss()
    optimizer_omega = torch.optim.Adam(self.critic.parameters(), lr=0.01)
    
    next_states = torch.Tensor(next_states)
    next_actions = self.actor(next_states)

    tmp = torch.cat([next_states, next_actions], dim=1)
    Q_omega = self.critic(tmp)
    delta = torch.Tensor(rewards).view(-1,1) + self.gamma * torch.mul(torch.Tensor(np.logical_not(dones)).view(-1,1), Q_omega)

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    tmp = torch.cat([states, actions], dim=1)
    Q_omega = self.critic(tmp)

    optimizer_omega.zero_grad()
    loss_omega = criterion_omega(Q_omega, delta)
    loss_omega.backward()
    optimizer_omega.step()
    
    # actor
    optimizer_theta = torch.optim.Adam(self.actor.parameters(), lr=0.01)
    
    actions = self.actor(states)
    
    optimizer_theta.zero_grad()
    loss_theta = torch.mean(-self.critic(torch.cat([states, actions], dim=1)))
    loss_theta.backward()
    optimizer_theta.step()
    
