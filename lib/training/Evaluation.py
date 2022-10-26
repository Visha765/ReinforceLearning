from tqdm import tqdm
import numpy as np
import sys, os
import gym
from ..model.agent import Agent


def Evaluation(env, agent, max_step: int, episode: int, seed: int):

  rewards = []

  for it in range(episode):
    state = env.reset()
    sum_reward = 0
    # for t in tqdm(range(max_step)):
    for t in range(max_step):
      # env.render()
      action = agent.select_action(state)
      next_state, reward, done, info = env.step(action)
      sum_reward += reward
      state = next_state
      if done:
        break  
    rewards.append(sum_reward)

  return rewards;
