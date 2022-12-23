import numpy as np
import sys, os
import re
import glob
import gym

from lib.training.train import Train
from lib.training.evaluate import Evaluate

# d is dataclass
def Worker(d):
    ## Training ###
    print('-'*10, "start Train", d.train_seed, '-'*10)
    
    env = gym.make(d.env_name)
    agent = d.agent()
    env.seed(d.train_seed)
    np.random.seed(d.train_seed)
    # save directory
    path = f"out/{d.dir_name}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)
    
    Train(env=env, agent=agent, end_step=d.train_step, interval=d.interval, path=path)
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", d.train_seed, '-'*10)
    
    # files = glob.glob(os.path.join(path, "*.pickle"))
    # files = [os.path.split(file)[1] for file in files]
    # files.sort()
    rewards_list = []
    for saved_step in range(0, d.train_step+1, d.interval):
    # for file in files:
        filename = f"log_step{saved_step}.pickle"
        env = gym.make(d.env_name)
        env.seed(d.eval_seed)
        agent = agent.load_models(path=path, filename=filename)
        rewards = Evaluate(env=env, agent=agent, max_step=d.eval_step, episode=d.episode)
        rewards_list.append(rewards) 
        env.close()
    return rewards_list