import numpy as np
import torch
import torch.nn as nn

from lib.util.custom_tanh import *

class ActorNet(nn.Module):
  def __init__(self, n, m):
    super(ActorNet, self).__init__()
    self.flatten = nn.Flatten()
    self.stack = nn.Sequential(
      nn.Linear(n, 256),
      nn.ReLU(),
      nn.Linear(256, m),
      Lambda(custom_tanh.apply)
    )

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y
    
    
class Actor():
  def __init__(self, n, m, sigma_lr=3*1e-4, target_tau=0.005, sigma_target=0.2, c=0.5) -> None:
    self.tau = [-2, 2]
        
    self.target_tau = target_tau
    self.c = c
    self.sigma_target = sigma_target
    
    self.net = ActorNet(n, m)
    self.net_target = ActorNet(n, m)
    self.optimizer = torch.optim.SGD(self.net.parameters(), lr=sigma_lr)
    
  def policy(self, states):
    return self.net(states)
  
  # Target Policy Smoothing Regularization
  def target_policy_sr(self, states):
    noises =  np.random.normal(0, self.sigma_target)
    noises = torch.tensor(noises).clip(-self.c, self.c).view(-1,1)
    policy_actions = self.net_target(states)
    policy_actions = (policy_actions + noises).clip(*self.tau)
    return policy_actions
  
  def loss_optimize(self, states, critic):
    self.optimizer.zero_grad()
    policy_actions = self.policy(states)
    Q = critic.estimate(states, policy_actions)
    loss = -Q.mean()
    loss.backward()
    self.optimizer.step()
    
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)