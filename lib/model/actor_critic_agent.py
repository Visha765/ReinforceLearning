import numpy as np
import sys, os
import pickle
import torch, torch.nn as nn
import copy

from lib.model.agent import Agent
from lib.model.replay_buffer import ReplayBuffer
from lib.model.actor import ActorNet
from lib.model.critic import CriticNet
from lib.util.xy2theta import xy2theta

class ActorCriticAgent(Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
    gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005 ,actor_interval=2, sigma_target=0.2, c=0.5):
    self.tau = (-2, 2)
    
    self.buffer = ReplayBuffer(buffer_size)
    self.batch_size = batch_size
    self.gamma = gamma
    self.sigma_beta = sigma_beta
    self.T_expl = T_expl
    self.target_tau = target_tau
    self.actor_interval = actor_interval
    self.sigma_target = sigma_target
    self.c = c
    
    self.actor = ActorNet(2, 1)
    self.actor_target = ActorNet(2, 1)
    self.critic1 = CriticNet(2, 1)
    self.critic2 = CriticNet(2, 1)
    self.critic1_target = CriticNet(2, 1)
    self.critic2_target = CriticNet(2, 1)
    
    self.criterion_critic = nn.MSELoss()
    self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=sigma_lr)
    self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=sigma_lr)
    self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=sigma_lr)
    

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
    return self.actor(self.list2tensor([state,])).detach().numpy()[0] #tensor -> ndarray
  
  def select_exploratory_action(self, state, current_step):
    if current_step < self.T_expl:
      return np.random.uniform(*self.tau, 1)
    
    D = (self.tau[1] - self.tau[0]) * self.sigma_beta/2
    return self.select_action(state) + np.random.normal(0, D**2)

  def train(self, state, action, next_state, reward, done, \
      current_step):
    state = xy2theta(state)
    next_state = xy2theta(next_state)
    self.buffer.add(state, action, next_state, reward, done)
    
    states, actions, next_states, rewards, dones = [self.list2tensor(i) for i in self.buffer.sample(self.batch_size)]
    
    # states = self.list2tensor(states)
    # actions = self.list2tensor(actions)
    # next_states = self.list2tensor(next_states)
    # rewards = self.list2tensor(rewards)
    # dones_rev = self.list2tensor(np.logical_not(dones))
    dones_rev = torch.tensor(list(map(lambda x: not x, dones)))
    
    ## critic    
    self.optimizer_critic1.zero_grad()
    self.optimizer_critic2.zero_grad()
    # Target Policy Smoothing Regularization
    noises =  np.random.normal(0, self.sigma_target)
    noises = self.list2tensor(np.clip(noises, -self.c, self.c)).view(-1,1)
    next_pred_actions = (self.actor_target(next_states) + noises).clip(*self.tau)
    
    # Clipped Double Q-Learning
    Q_actor1 = self.critic1_target(torch.cat([next_states, next_pred_actions], dim=1))
    Q_actor2 = self.critic2_target(torch.cat([next_states, next_pred_actions], dim=1))
    Q_min = torch.tensor(list(map(lambda q1, q2: min(q1,q2), Q_actor1, Q_actor2)))
    delta = rewards.view(-1,1) \
      + self.gamma * torch.mul(dones_rev.view(-1,1), Q_min.view(-1,1))
      
    Q1 = self.critic1(torch.cat([states, actions], dim=1))
    Q2 = self.critic2(torch.cat([states, actions], dim=1))
    
    loss_omega1 = self.criterion_critic(delta, Q1)    
    loss_omega1.backward(retain_graph=True)
    self.optimizer_critic1.step()
    
    loss_omega2 = self.criterion_critic(delta, Q2)
    loss_omega2.backward(retain_graph=True)
    self.optimizer_critic2.step()
    
    
    ## actor    
    # Delayed Policy Update
    if (current_step//self.actor_interval == 0): 
      self.optimizer_actor.zero_grad()
      pred_actions = self.actor(states)
      loss_theta = (-self.critic1(torch.cat([states, pred_actions], dim=1))).mean()
      loss_theta.backward()
      self.optimizer_actor.step()
      
      for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
            
      for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
      for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)
    
    
  def list2tensor(self, x):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return torch.Tensor(x)
    
