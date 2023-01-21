import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import copy

from lib.util.custom_tanh import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('loss', 'step'))

class ActorNet(nn.Module):
  def __init__(self, dim_state, dim_action, hidden1_size=256):
    super(ActorNet, self).__init__()
    self.flatten = nn.Flatten()
    self.stack = nn.Sequential(
      nn.Linear(dim_state, hidden1_size),
      nn.ReLU(),
      nn.Linear(hidden1_size, dim_action),
      Lambda(custom_tanh.apply)
    )
    
    for m in self.modules():
      if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #     # nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y
    
class Actor():
  def __init__(self, dim_state, dim_action, lr=3*1e-4, target_tau=0.005, sigma_sr=0.2, c=0.5, interval=500):
    self.tau = (-2, 2)
        
    self.target_tau = target_tau
    self.c = c
    self.sigma_sr = sigma_sr
    self.interval = interval
    
    self.net = ActorNet(dim_state, dim_action).to(device)
    self.net_target = copy.deepcopy(self.net).to(device)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    self.losses = []
    
  def policy(self, states, mode='n'):
    net = (self.net if mode=='n' else self.net_target).to(device)
    return net(states)
  
  def policy_sr(self, states, mode='n'):
    noises = torch.normal(0, self.sigma_sr, (1,)).clip(-self.c, self.c).view(-1,1).to(device)
    actions = self.policy(states, mode=mode)
    return (actions + noises).clip(*self.tau)

  def loss_optimize(self, states, critic, current_step):
    actions = self.policy(states)
    loss = -critic.estimate(states, actions).mean()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    if current_step % self.interval == 0:
      self.losses.append(Transition(loss=loss.item(), step=current_step))
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
