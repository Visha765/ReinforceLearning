import torch

from lib.model.actor_critic_agent import ActorCriticAgent
from lib.model.critic import Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent(ActorCriticAgent):
  def __init__(self, env, buffer_size, batch_size, lr=3*1e-4, \
      gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    super().__init__(env, buffer_size, batch_size, lr=lr, \
        gamma=gamma, sigma_beta=sigma_beta, T_expl=T_expl, target_tau=target_tau, actor_interval=actor_interval, sigma_sr=sigma_sr, c=c)
    self.critic2 = Critic(self.dim_state, self.dim_action, lr=lr, target_tau=target_tau)

  def transform_state(self, state):
    return state

  def train(self, state, action, next_state, reward, done, current_step):
    self.buffer.add(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()
    
    #Target Policy Smoothing Regularization
    next_policy_actions = self.actor.policy_sr(next_states, mode='t')
    # Clipped Double Q-Learning
    Q1 = self.critic.estimate(next_states, next_policy_actions, mode='t')
    Q2 = self.critic2.estimate(next_states, next_policy_actions, mode='t')
    Q_min = torch.minimum(Q1, Q2)
    delta = self.delta(Q_min, rewards, dones_rev, self.gamma)
      
    self.critic.loss_optimize(states, actions, delta, current_step)
    self.critic2.loss_optimize(states, actions, delta, current_step)
    
    # Delayed Policy Update
    if (current_step % self.actor_interval == 0): 
      self.actor.loss_optimize(states, self.critic, current_step)
      # Target Actor & Target Critic
      self.actor.update_target_params()
      self.critic.update_target_params()
      self.critic2.update_target_params()