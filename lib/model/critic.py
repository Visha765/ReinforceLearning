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
      nn.Linear(hidden2_size, 1)
    )
    
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y

class Critic():
  def __init__(self, dim_state, dim_action, lr=3*1e-4, target_tau=0.005, interval=1000):
    self.target_tau = target_tau
    self.interval = interval
    
    self.net = CriticNet(dim_state, dim_action).to(device)
    self.net_target = copy.deepcopy(self.net).to(device)
    self.criterion = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-2)
    
    self.losses = []
    
  def estimate(self, states, actions, mode='n'):
    net = (self.net if mode=='n' else self.net_target).to(device)
    x = torch.cat([states, actions], dim=1)
    return net(x)
  
  def loss_optimize(self, states, actions, delta, current_step):
    Q = self.estimate(states, actions)
    loss = self.criterion(Q, delta)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    if current_step % self.interval == 0:
      self.losses.append(Transition(loss=loss.item(), step=current_step))
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
  
  @classmethod
  def delta(cls, Q, rewards, dones_rev, gamma = 0.99):
    with torch.no_grad():
      delta = rewards.view(-1,1) + dones_rev.view(-1,1) * Q.view(-1,1) * gamma
      return delta