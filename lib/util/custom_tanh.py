import torch, torch.nn as nn
import gym

from lib.util.list2tensor import list2tensor

env_name = "Pendulum-v0"
# env_name = "BipedalWalker-v3"
env = gym.make(env_name)
a_min = list2tensor(env.action_space.low)
a_max = list2tensor(env.action_space.high)

def d_tanh(x): 
    return 1 / (x.cosh() ** 2)

class custom_tanh(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = (a_max + a_min) / 2 + (a_max - a_min) / 2 * x.tanh()
        return y

    @staticmethod
    def backward(ctx, dL_dy):
        x, = ctx.saved_tensors
        dy_dx = (a_max - a_min) / 2 * d_tanh(x)
        dL_dx = dL_dy * dy_dx
        return dL_dx

# wrapper
class wrapper(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): 
        return self.func(x)