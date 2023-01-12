import numpy as np
import torch
import torch.nn as nn
import copy

from lib.util.custom_tanh import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y
    
class Actor():
  def __init__(self, dim_state, dim_action, sigma_lr=3*1e-4, target_tau=0.005, sigma_sr=0.2, c=0.5) -> None:
    self.tau = [-2, 2]
        
    self.target_tau = target_tau
    self.c = c
    self.sigma_sr = sigma_sr
    
    self.net = ActorNet(dim_state, dim_action).to(device)
    self.net_target = copy.deepcopy(self.net).to(device)
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=sigma_lr)
    
  def policy(self, states):
    return self.net(states)
  
  def target_policy(self, states):
    return self.net_target(states)
  
  # Policy Smoothing Regularization
  def policy_sr(self, states):
    noises =  np.random.normal(0, self.sigma_sr)
    noises = torch.tensor(noises).clip(-self.c, self.c).view(-1,1).to(device)
    policy_actions = self.net_target(states)
    policy_actions = (policy_actions + noises).clip(*self.tau)
    return policy_actions
  # Target Policy Smoothing Regularization
  def target_policy_sr(self, states):
    noises =  np.random.normal(0, self.sigma_sr)
    noises = torch.tensor(noises).clip(-self.c, self.c).view(-1,1).to(device)
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