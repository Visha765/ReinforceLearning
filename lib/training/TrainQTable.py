from genericpath import isfile
from tqdm import tqdm
import numpy as np
import sys, os
import glob
import gym

from lib.util.fetchPikle import fetch_pikles
from lib.model.agent import Agent

def Train(Agent: Agent, env_name: str, step: int, seed: int, interval: int, path):
    
  env = gym.make(env_name)
  env.seed(seed)
  env.action_space.seed(seed)
  agent = Agent(env, 8, 8, 0.01)
  
  state = env.reset()
  current_step = 0
  
  ## load model if exists
  # saved_steps, files =fetch_pikles(path)
  # if len(saved_steps)!=0:
  #   env, agent, current_step, info = agent.load_models(path, files[-1])
  
  print('-'*10, "start Train", '-'*10)
  for t in tqdm(range(current_step, step)):
    # env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    
    # ## save model per interval
    # if (t % interval == 0 and t!= 0):
    #   agent.save_models(env=env,
    #                     current_step=t, 
    #                     seed=seed, 
    #                     info=info, 
    #                     path=path
    #                   )
      
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
      state = env.reset()


  # agent.save_models(env=env,
  #                   current_step=step, 
  #                   seed=seed, 
  #                   info=info, 
  #                   path=path
  #                 )
  env.close()

  return agent