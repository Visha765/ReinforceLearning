import numpy as np
import sys, os
import glob
import gym

from lib.training.train import Train
from lib.training.evaluate import Evaluate

# d is dataclass
def Worker(d):
    ## Training ###
    print('-'*10, "start Train", '-'*10)
    
    env = gym.make(d.env_name)
    agent = d.agent()
    env.seed(d.train_seed)
    np.random.seed(d.train_seed)
    # save directory
    path = f"out/{d.env_name}_{d.agent_name}_seed{d.train_seed}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)
    
    Train(env=env, agent=agent, end_step=d.train_step, interval=d.interval, path=path)
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", '-'*10)
    
    files = glob.glob(os.path.join(path, "*.pickle"))
    files = [os.path.split(file)[1] for file in files]
    rewards_list = [[] for i in range(d.train_step//d.interval+1)]
    for file in files:
        env = gym.make(d.env_name)
        env.seed(d.eval_seed)
        agent, saved_step = agent.load_models(path=path, filename=file)
        if saved_step > d.train_step: 
            break
        rewards = Evaluate(env=env, agent=agent, max_step=d.eval_step, episode=d.episode)
        rewards_list[saved_step//d.interval].extend(rewards) 
        env.close()
    return rewards_list