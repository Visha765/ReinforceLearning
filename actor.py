import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_tanh import *

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
    
    def update(state, action):
      pass
    