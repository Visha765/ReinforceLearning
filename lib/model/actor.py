import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
  def __init__(self, sigma_lr=3*1e-4, target_tau=0.005, sigma_target=0.2, c=0.5) -> None:
    self.tau = [-2, 2]
    
    self.target_tau = target_tau
    self.c = c
    self.sigma_target = sigma_target
  
    
    self.net = ActorNet(2, 1)
    self.net_target = ActorNet(2, 1)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=sigma_lr)
    
  def predict(self, states):
    return self.net(states)
  
  # Target Policy Smoothing Regularization
  def pred_actions(self, states):
    noises =  np.random.normal(0, self.sigma_target)
    # noises = self.list2tensor(np.clip(noises, -self.c, self.c)).view(-1,1)
    noises = torch.tensor(noises).clip(-self.c, self.c).view(-1,1)
    pred_actions = self.net_target(states)
    pred_actions = (pred_actions + noises).clip(*self.tau)
    return pred_actions
  
  def loss_optimize(self, states, critic):
    self.optimizer.zero_grad()
    pred_actions = self.predict(states)
    # Q_1 = self.critic1(torch.cat([states, pred_actions], dim=1))
    Q_1 = critic.estimate(states, pred_actions)
    loss = (-Q_1).mean()
    loss.backward()
    self.optimizer.step()
    
  def update_target_params(self):
    for target_param, param in zip(self.net_target.parameters(), self.net.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)