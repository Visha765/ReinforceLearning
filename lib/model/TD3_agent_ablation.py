import torch

from lib.model.critic import Critic
from lib.model.TD3_agent import TD3Agent

# Agent without Target Actor & Target Critic
class TD3Agent_withoutTATC(TD3Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
      gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    super().__init__(buffer_size, batch_size, sigma_lr=sigma_lr, \
        gamma=gamma, sigma_beta=sigma_beta, T_expl=T_expl, target_tau=target_tau, actor_interval=actor_interval, sigma_sr=sigma_sr, c=c)
    
  def train(self, state, action, next_state, reward, done, current_step):
    self.add_buffer(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()
    
    #Target Policy Smoothing Regularization
    next_policy_actions = self.actor.policy_sr(next_states).to(self.device)
    # Clipped Double Q-Learning
    Q1 = self.critic1.estimate(next_states, next_policy_actions)
    Q2 = self.critic2.estimate(next_states, next_policy_actions)
    Q_min = torch.minimum(Q1, Q2)
    delta = Critic.delta(Q_min, rewards, dones_rev, self.gamma)
      
    self.critic1.loss_optimize(states, actions, delta)
    self.critic2.loss_optimize(states, actions, delta)
    
    # Delayed Policy Update
    if (current_step % self.actor_interval == 0): 
      self.actor.loss_optimize(states, self.critic1)


# Agent without Target Policy Smoothing Regularization
class TD3Agent_withoutTPSR(TD3Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
      gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    super().__init__(buffer_size, batch_size, sigma_lr=sigma_lr, \
        gamma=gamma, sigma_beta=sigma_beta, T_expl=T_expl, target_tau=target_tau, actor_interval=actor_interval, sigma_sr=sigma_sr, c=c)
    
  def train(self, state, action, next_state, reward, done, current_step):
    self.add_buffer(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()
    
    #without Target Policy Smoothing Regularization
    next_policy_actions = self.actor.target_policy(next_states).to(self.device)
    # Clipped Double Q-Learning
    Q1 = self.critic1.target_estimate(next_states, next_policy_actions)
    Q2 = self.critic2.target_estimate(next_states, next_policy_actions)
    Q_min = torch.minimum(Q1, Q2)
    delta = Critic.delta(Q_min, rewards, dones_rev, self.gamma)
      
    self.critic1.loss_optimize(states, actions, delta)
    self.critic2.loss_optimize(states, actions, delta)
    
    # Delayed Policy Update
    if (current_step % self.actor_interval == 0): 
      self.actor.loss_optimize(states, self.critic1)
      # Target Actor & Target Critic
      self.actor.update_target_params()
      self.critic1.update_target_params()
      self.critic2.update_target_params()


# Agent without Delayed Policy Update 
class TD3Agent_withoutDPU(TD3Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
      gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    super().__init__(buffer_size, batch_size, sigma_lr=sigma_lr, \
        gamma=gamma, sigma_beta=sigma_beta, T_expl=T_expl, target_tau=target_tau, actor_interval=actor_interval, sigma_sr=sigma_sr, c=c)
    
  def train(self, state, action, next_state, reward, done, current_step):
    self.add_buffer(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()
    
    #Target Policy Smoothing Regularization
    next_policy_actions = self.actor.target_policy_sr(next_states).to(self.device)
    # Clipped Double Q-Learning
    Q1 = self.critic1.target_estimate(next_states, next_policy_actions)
    Q2 = self.critic2.target_estimate(next_states, next_policy_actions)
    Q_min = torch.minimum(Q1, Q2)
    delta = Critic.delta(Q_min, rewards, dones_rev, self.gamma)
      
    self.critic1.loss_optimize(states, actions, delta)
    self.critic2.loss_optimize(states, actions, delta)
    
    # without Delayed Policy Update 
    self.actor.loss_optimize(states, self.critic1)
    # Target Actor & Target Critic
    self.actor.update_target_params()
    self.critic1.update_target_params()
    self.critic2.update_target_params()


# Agent without Clipped Double Q-Learning
class TD3Agent_withoutCDQ(TD3Agent):
  def __init__(self, buffer_size, batch_size, sigma_lr=3*1e-4, \
      gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    super().__init__(buffer_size, batch_size, sigma_lr=sigma_lr, \
        gamma=gamma, sigma_beta=sigma_beta, T_expl=T_expl, target_tau=target_tau, actor_interval=actor_interval, sigma_sr=sigma_sr, c=c)
    
  def train(self, state, action, next_state, reward, done, current_step):
    self.add_buffer(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()
    
    #none of Target Policy Smoothing Regularization
    next_policy_actions = self.actor.target_policy_sr(next_states).to(self.device)
    # without Clipped Double Q-Learning
    Q = self.critic1.target_estimate(next_states, next_policy_actions)
    delta = Critic.delta(Q, rewards, dones_rev, self.gamma)
      
    self.critic1.loss_optimize(states, actions, delta)
    
    # Delayed Policy Update
    if (current_step % self.actor_interval == 0): 
      self.actor.loss_optimize(states, self.critic1)
      # Target Actor & Target Critic
      self.actor.update_target_params()
      self.critic1.update_target_params()
