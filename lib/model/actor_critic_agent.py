import numpy as np
import os
import torch

from lib.model.agent import Agent
from lib.model.replay_buffer import ReplayBuffer
from lib.model.actor import Actor
from lib.model.critic import Critic
from lib.util.loss_plot import LossPlot
from lib.util.list2tensor import list2tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCriticAgent(Agent):
  def __init__(self, env, buffer_size, batch_size, lr=3*1e-4, \
      gamma=0.99, sigma_beta=0.1, T_expl=10000, target_tau=0.005, actor_interval=2, sigma_sr=0.2, c=0.5):
    self.action_space = env.action_space
    self.dim_state = env.observation_space.shape[0]
    self.dim_action = env.action_space.shape[0]

    self.actor_interval = actor_interval
    self.sigma_beta = sigma_beta
    self.T_expl = T_expl
    self.gamma = gamma
    
    self.batch_size = batch_size
    self.buffer = ReplayBuffer(buffer_size)
    
    self.actor = Actor(self.dim_state, self.dim_action, self.action_space, lr=lr, target_tau=target_tau, sigma_sr=sigma_sr, c=c)
    self.critic = Critic(self.dim_state, self.dim_action, lr=lr, target_tau=target_tau)  

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
    states = list2tensor([state,])
    return self.actor.policy(states)[0].detach().cpu().numpy() #tensor -> ndarray
  
  def select_exploratory_action(self, state, current_step):
    if current_step < self.T_expl:
      return np.random.uniform(self.action_space.low, self.action_space.high)
    
    action = self.select_action(state)
    D = (self.action_space.high - self.action_space.low) / 2 * self.sigma_beta
    noise = np.random.normal(0, D**2)
    return np.clip(action + noise, self.action_space.low, self.action_space.high)
  
  def train(self, state, action, next_state, reward, done, current_step):
    self.buffer.add(state, action, next_state, reward, done)
    if (len(self.buffer)) < self.batch_size: return None # skip
    states, actions, next_states, rewards, dones_rev = self.sample_buffer()

    next_policy_actions = self.actor.policy(next_states)
    Q = self.critic.estimate(next_states, next_policy_actions)
    delta = Critic.delta(Q, rewards, dones_rev, self.gamma)
    self.critic.loss_optimize(states, actions, delta, current_step)
    self.actor.loss_optimize(states, self.critic, current_step)
    
  def delta(self, Q, rewards, dones_rev, gamma):
    with torch.no_grad():
      delta = rewards.view(-1, 1) + dones_rev.view(-1, 1) * Q.view(-1, 1) * gamma
      return delta
    
  def sample_buffer(self):
    if (len(self.buffer)) < self.batch_size: raise
    states, actions, next_states, rewards, dones = [list2tensor(lst) for lst in self.buffer.sample(self.batch_size)]
    dones_rev = list2tensor(list(map(lambda x: not x, dones)))  
    return states, actions, next_states, rewards, dones_rev
  
  def plot(self, model, path):
    losses = [i.loss for i in model.losses]
    steps = [i.step for i in model.losses]
    filename = f"{model.__class__.__name__}_loss"
    LossPlot(losses, steps, filename, path)
    
  def plot_loss(self, path):
    self.plot(self.actor, path)
    self.plot(self.critic, path)
