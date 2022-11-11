from multiprocessing import Pool
import numpy as np
import sys, os
import gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.train import Train
from lib.training.evaluation import Evaluation
from lib.util.fetchPickle import fetch_pickle
from lib.util.linePlot import LinePlot
from lib.model.qTableAgent import QTableAgent


### Condition ###
env_name = 'Pendulum-v0'
    
train_step = 500000
train_seeds = [11, 13, 17, 19, 23]
interval = 1000
K, L = 10, 9

episode = 10 # 10
eval_step = 10000
eval_seed = 0
    

def thread(train_seed):
    ### Train ###
    print('-'*10, "start Train", '-'*10)
    
    env = gym.make(env_name)
    env.seed(train_seed)
    np.random.seed(train_seed)
    agent = QTableAgent(K, L)
    path=f"out/{agent.__class__.__name__ }_seed{train_seed}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)

    Train(env=env, agent=agent, end_step=train_step, seed=train_seed, save_interval=interval, path=path)
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", '-'*10)
    
    saved_steps, files = fetch_pickle(path)
    data_list = []
    for file in files:
        env = gym.make(env_name)
        env.seed(eval_seed)
        agent, _ = agent.load_models(path=path, filename=file)
        rewards = Evaluation(env=env, agent=agent, max_step=eval_step, episode=episode, seed=eval_seed)
        data_list.append(rewards)
        env.close()
        
    ### Visualize ###
    LinePlot(data_list=data_list, label_list=saved_steps, env_name=env_name, seed=train_seed, path=path)


def main():
    p = Pool(len(train_seeds))
    p.map(thread, train_seeds)
    p.close()
    
if __name__ == '__main__':
    main()
    
