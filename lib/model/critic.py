import numpy as np
import torch
import torch.nn as nn

from lib.util.custom_tanh import *


class CriticNet(nn.Module):
  def __init__(self, dim_state, dim_action, hidden1_size=256, hidden2_size=256):
    super(CriticNet, self).__init__()
    self.flatten = nn.Flatten()
    self.stack = nn.Sequential(
      nn.Linear(dim_state+dim_action, hidden1_size),
      nn.ReLU(),
      nn.Linear(hidden1_size, hidden2_size),
      nn.ReLU(),
      nn.Linear(hidden2_size, 1),
      Lambda(custom_tanh.apply),
    )

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y

class Critic():
  def __init__(self, dim_state, dim_action, sigma_lr=3*1e-4, target_tau=0.005) -> None:
    self.target_tau = target_tau
    
    self.net = CriticNet(dim_state, dim_action)
    self.net_target = CriticNet(dim_state, dim_action)
    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=sigma_lr)
    
  def estimate(self, states, actions):
    x = torch.cat([states, actions], dim=1)
    return self.net(x)
  
  def target_estimate(self, states, actions):
    x = torch.cat([states, actions], dim=1)
    return self.net_target(x)
  
  def loss_optimize(self, states, actions, delta):
    self.optimizer.zero_grad()
    Q = self.estimate(states, actions)
    loss = self.criterion(delta, Q)
    loss.backward()
    self.optimizer.step()
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
  
  @classmethod
  def delta(cls, Q1, Q2, rewards, dones_rev, gamma = 0.99):
    with torch.no_grad():
      # Q_min = torch.tensor(list(map(lambda q1, q2: min(q1,q2), Q1, Q2)))
      Q_min = torch.minimum(Q1, Q2)
      delta = rewards.view(-1,1) \
        + torch.mul(dones_rev.view(-1,1), Q_min.view(-1,1)) * gamma
      return delta