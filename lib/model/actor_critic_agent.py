import numpy as np
import sys, os
import pickle
import torch, torch.nn as nn
import copy

from lib.model.agent import Agent
from lib.model.replay_buffer import ReplayBuffer
from lib.model.actor import Actor
from lib.model.critic import Critic
from lib.util.xy2theta import xy2theta

class ActorCriticAgent(Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
    gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    self.tau = (-2, 2)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    self.actor_interval = actor_interval
    self.sigma_beta = sigma_beta
    self.T_expl = T_expl
    self.gamma = gamma
    
    self.batch_size = batch_size
    self.buffer = ReplayBuffer(buffer_size)
    dim_state, dim_action = 2, 1
    self.actor = Actor(dim_state, dim_action, sigma_lr=sigma_lr, target_tau=target_tau, sigma_sr=sigma_sr, c=c)
    self.critic1 = Critic(dim_state, dim_action, sigma_lr=sigma_lr, target_tau=target_tau)
    self.critic2 = Critic(dim_state, dim_action, sigma_lr=sigma_lr, target_tau=target_tau)
    

  def save_models(self, current_step, path):
    # filename = "log_step{:07d}.pickle".format(current_step)
    filename = f"log_step{current_step}.pickle"
    tmp = copy.deepcopy(self.buffer)
    self.buffer = []
    with open(os.path.join(path, filename), 'wb') as f:
      pickle.dump(self, f)
    self.buffer = tmp

  def load_models(self, path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
      data = pickle.load(f)
    return data

  def select_action(self, state):
    state = xy2theta(state)
    states = self.list2tensor([state,])
    return self.actor.policy(states).detach().numpy()[0] #tensor -> ndarray
  
  def select_exploratory_action(self, state, current_step):
    if current_step < self.T_expl:
      return np.random.uniform(*self.tau, 1)
    
    D = (self.tau[1] - self.tau[0]) * self.sigma_beta/2
    return self.select_action(state) + np.random.normal(0, D**2)

  def train(self, state, action, next_state, reward, done, current_step):
    # add exp into buffer
    state = xy2theta(state)
    next_state = xy2theta(next_state)
    self.buffer.add(state, action, next_state, reward, done)
    
    if (len(self.buffer)) < self.batch_size:
      return 
    
    # sample exp arrays from buffer
    states, actions, next_states, rewards, dones = [self.list2tensor(lst) for lst in self.buffer.sample(self.batch_size)]
    dones_rev = torch.tensor(list(map(lambda x: not x, dones)), device=self.device, dtype=torch.float32)


    #Target Policy Smoothing Regularization
    next_policy_actions = self.actor.target_policy_sr(next_states).to(self.device)
    # Clipped Double Q-Learning
    Q1 = self.critic1.target_estimate(next_states, next_policy_actions)
    Q2 = self.critic2.target_estimate(next_states, next_policy_actions)
    delta = Critic.delta(Q1, Q2, rewards, dones_rev, self.gamma)
      
    self.critic1.loss_optimize(states, actions, delta)
    self.critic2.loss_optimize(states, actions, delta)
    
    # Delayed Policy Update
    if (current_step % self.actor_interval == 0): 
      self.actor.loss_optimize(states, self.critic1)
      # Target Actor & Target Critic
      self.actor.update_target_params()
      self.critic1.update_target_params()
      self.critic2.update_target_params()
    
  def list2tensor(self, x):
    if not isinstance(x, np.ndarray):
      x = np.array(x, dtype=np.float32)
    return torch.Tensor(x, device=self.device)
    
