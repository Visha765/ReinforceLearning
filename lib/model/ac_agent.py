import numpy as np
import os
import torch

from lib.model.agent import Agent
from lib.model.replay_buffer import ReplayBuffer
from lib.model.actor import Actor
from lib.model.critic import Critic
from lib.util.xy2theta import xy2theta
from lib.util.loss_plot import LossPlot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ACAgent(Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
    gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    self.tau = (-2, 2)

    self.actor_interval = actor_interval
    self.sigma_beta = sigma_beta
    self.T_expl = T_expl
    self.gamma = gamma
    
    self.batch_size = batch_size
    self.buffer = ReplayBuffer(buffer_size)
    dim_state, dim_action = 2, 1
    self.actor = Actor(dim_state, dim_action, sigma_lr=sigma_lr, target_tau=target_tau, sigma_sr=sigma_sr, c=c)
    self.critic = Critic(dim_state, dim_action, sigma_lr=sigma_lr, target_tau=target_tau)
    

  def save_models(self, current_step, path):
    filename = f"log_step{current_step}.pth"
    model_path = os.path.join(path, filename)
    model = self.actor.net
    torch.save(model.to('cpu').state_dict(), model_path)

  def load_models(self, path, saved_step):
    filename = f"log_step{saved_step}.pth"
    model_path = os.path.join(path, filename)
    model = self.actor.net
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

  def select_action(self, state):
    state = xy2theta(state)
    states = self.list2tensor([state,])
    return self.actor.policy(states)[0].detach().cpu().numpy() #tensor -> ndarray
  
  def select_exploratory_action(self, state, current_step):
    if current_step < self.T_expl:
      return np.random.uniform(*self.tau, 1)
    
    action = self.select_action(state)
    D = (self.tau[1] - self.tau[0]) * self.sigma_beta/2
    noise = np.random.normal(0, D**2)
    return np.clip(action + noise, *self.tau)

  def train(self, state, action, next_state, reward, done, current_step):
    self.add_buffer(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()

    next_policy_actions = self.actor.policy(next_states)
    Q = self.critic.estimate(next_states, next_policy_actions)
    delta = Critic.delta(Q, rewards, dones_rev, self.gamma)
    self.critic.loss_optimize(states, actions, delta, current_step)
    self.actor.loss_optimize(states, self.critic, current_step)
    
  
  def list2tensor(self, x):
    x = np.array(x, dtype=np.float32)
    return torch.Tensor(x).to(device)
  
  def add_buffer(self, state, action, next_state, reward, done):
    # add exp into buffer
    state = xy2theta(state)
    next_state = xy2theta(next_state)
    self.buffer.add(state, action, next_state, reward, done)
    
  def sample_buffer(self):
    # sample exp arrays from buffer
    if (len(self.buffer)) < self.batch_size: raise
    states, actions, next_states, rewards, dones = [self.list2tensor(lst) for lst in self.buffer.sample(self.batch_size)]
    dones_rev = self.list2tensor(list(map(lambda x: not x, dones)))  
    return states, actions, next_states, rewards, dones_rev
  
  def plot(self, model, path):
    losses = [i.loss for i in model.losses]
    steps = [i.step for i in model.losses]
    LossPlot(losses, steps, model.__class__.__name__, path)
    
  def plot_loss(self, path):
    self.plot(self.actor, path)
    self.plot(self.critic, path)
    

