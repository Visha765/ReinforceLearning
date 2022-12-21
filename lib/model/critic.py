import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_tanh import *


class CriticNet(nn.Module):
  def __init__(self, n, m):
    super(CriticNet, self).__init__()
    self.flatten = nn.Flatten()
    self.stack = nn.Sequential(
      nn.Linear(n + m, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      Lambda(custom_tanh.apply),
    )

  def forward(self, x):
    x = self.flatten(x)
    y = self.stack(x)
    return y

  # def update():
  #   next_states = torch.tensor(next_states)
  #   next_actions = self.actor(next_states)
  #   Q_omega = self.critic(torch.cat([next_states, next_actions], dim=1))
  #   delta = rewards + self.gamma * torch.mul(np.logical_not(dones), Q_omega)


def xy2theta(self, state):
    cos, sin = state[0:2]
    theta = np.arccos(cos) if sin > 0 else -np.arccos(cos)
    return (theta, state[2])
