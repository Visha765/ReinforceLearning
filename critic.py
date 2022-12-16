import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_tanh import *

class CriticNet(nn.Module):
    
    def __init__(self, n, m):
        super(CriticNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n+m, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            Lambda(custom_tanh.apply)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y
    
    # def backward(self, L):
    #     dL_dx = self.linear_relu_stack(L)
    #     return L
    
# class Q_omega():
     # state: (cos(theta), sin(theta), omega) --> (theta, omega)
def xy2theta(self, state):
    cos, sin = state[0:2]
    theta = np.arccos(cos) if sin > 0 else -np.arccos(cos)
    return (theta, state[2])