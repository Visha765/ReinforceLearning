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
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y
    
class Actor():
  def __init__(self, dim_state, dim_action, sigma_lr=3*1e-4, target_tau=0.005, sigma_sr=0.2, c=0.5) -> None:
    self.tau = (-2, 2)
        
    self.target_tau = target_tau
    self.c = c
    self.sigma_sr = sigma_sr
    
    self.net = ActorNet(dim_state, dim_action).to(device)
    self.net_target = copy.deepcopy(self.net).to(device)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=sigma_lr)
    
    self.losses = []
    
  def policy(self, states, mode='n'):
    net = (self.net if mode!='t' else self.net_target).to(device)
    return net(states)
  
  def policy_sr(self, states, mode='n'):
    net = (self.net if mode!='t' else self.net_target).to(device)
    noises = torch.normal(0, self.sigma_sr, (1,)).clip(-self.c, self.c).view(-1,1).to(device)
    actions = net(states)
    return (actions + noises).clip(*self.tau)

  def loss_optimize(self, states, critic, current_step):
    self.optimizer.zero_grad()
    policy_actions = self.policy(states)
    Q = critic.estimate(states, policy_actions)
    loss = -Q.mean()
    self.losses.append(Transition(loss.item(), current_step))
    loss.backward()
    self.optimizer.step()
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
