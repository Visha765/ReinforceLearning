import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import copy

from lib.util.custom_tanh import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('loss', 'step'))

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
      # Lambda(custom_tanh.apply),
    )
    
    for m in self.modules():
      if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y

class Critic():
  def __init__(self, dim_state, dim_action, sigma_lr=3*1e-4, target_tau=0.005) -> None:
    self.target_tau = target_tau
    
    self.net = CriticNet(dim_state, dim_action).to(device)
    self.net_target = copy.deepcopy(self.net).to(device)
    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=sigma_lr)
    
    self.losses = []
    
  def estimate(self, states, actions, mode='n'):
    net = self.net if mode!='t' else self.net_target
    x = torch.cat([states, actions], dim=1)
    return net(x)
  
  # def target_estimate(self, states, actions):
  #   x = torch.cat([states, actions], dim=1)
  #   return self.net_target(x)
  
  def loss_optimize(self, states, actions, delta, current_step):
    self.optimizer.zero_grad()
    Q = self.estimate(states, actions)
    loss = self.criterion(delta, Q)
    self.losses.append(Transition(loss.item(), current_step))
    loss.backward()
    self.optimizer.step()
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
  
  @classmethod
  def delta(cls, Q, rewards, dones_rev, gamma = 0.99):
    with torch.no_grad():
      delta = rewards.view(-1,1) \
        + torch.mul(dones_rev.view(-1,1), Q.view(-1,1)) * gamma
      return delta