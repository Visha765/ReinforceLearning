import numpy as np
import random
import sys, os
import torch
import gym
from tqdm import tqdm

from lib.training.train import Train
from lib.training.evaluate import Evaluate

# d is dataclass
def Worker(d):
    ## Training ###
    print('-'*10, "start Train", d.agent_name, d.train_seed, '-'*10)
    
    env = gym.make(d.env_name)
    agent = d.agent()
    env.seed(d.train_seed)
    np.random.seed(d.train_seed)
    random.seed(d.train_seed)
    torch.manual_seed(d.train_seed)
    torch.cuda.manual_seed(d.train_seed)
    
    # save directory
    path = f"out/{d.dir_name}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)
    
    Train(env=env, agent=agent, end_step=d.train_step, interval=d.interval, path=path)
    agent.plot_loss(interval=d.interval//100, path=path)
    env.close()
    

    ### Evaluation ###
    print('-'*10, "start Evaluation", d.agent_name, d.train_seed, '-'*10)

    rewards_list = []
    for saved_step in tqdm(range(0, d.train_step+1, d.interval)):
        env = gym.make(d.env_name)
        env.seed(d.eval_seed)
        agent.load_models(path=path, saved_step=saved_step)
        rewards = Evaluate(env=env, agent=agent, max_step=d.eval_step, episode=d.episode)
        rewards_list.append(rewards) 
        env.close()
    return rewards_list