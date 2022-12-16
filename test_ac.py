
import gym
from actor_critic_agent import ActorCriticAgent
from tqdm import tqdm
import numpy as np


max_step = 10000
episode = 5

env = gym.make('Pendulum-v0') 
agent = ActorCriticAgent(max_step, 32)

agents = []

state = env.reset()
for t in tqdm(range(max_step)):
  action = agent.select_exploratory_action(state, t)
  next_state, reward, done, info = env.step(action) 
  agent.train(state, action, next_state, reward, done) 
  state = next_state
  if done:
    state = env.reset() 
    
  if ((t+1) % 100 == 0):
    agents.append(agent)
env.close()

# agent = ActorCriticAgent(max_step, 32)
reward_agent = []
for agent in agents:
  
  env = gym.make('Pendulum-v0') 
  rewards = []
  for it in range(episode):
    state = env.reset()
    sum_reward = 0
    for t in range(max_step):
      # env.render()
      action = agent.select_action(state)
      next_state, reward, done, info = env.step(action)
      sum_reward += reward
      state = next_state
      if done:
        break 
    rewards.append(sum_reward)
  reward_agent.append(np.mean(rewards))
  env.close
  
  
print(reward_agent)
