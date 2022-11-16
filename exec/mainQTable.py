from multiprocessing import Pool
import numpy as np
import sys, os
import glob
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
interval = 10000
K, L = 10, 9

episode = 10 # 10
eval_step = 10000 #maxstep
eval_seed = 0


saved_steps = [i for i in range(interval, train_step+1, interval)]

def worker(train_seed):
    ### Train ###
    print('-'*10, "start Train", '-'*10)
    
    env = gym.make(env_name)
    env.seed(train_seed)
    np.random.seed(train_seed)
    agent = QTableAgent(K, L)
    
    path = f"out/{env_name}_{agent.__class__.__name__}_seed{train_seed}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)

    Train(env=env, agent=agent, end_step=train_step, seed=train_seed, save_interval=interval, path=path)
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", '-'*10)
    
    # saved_steps, files = fetch_pickle(path)
    files = glob.glob(os.path.join(path, "*.pickle"))
    files = [os.path.split(file)[1] for file in files]
    data = [[] for i in range(len(saved_steps))]
    for file in files:
        env = gym.make(env_name)
        env.seed(eval_seed)
        agent, saved_step= agent.load_models(path=path, filename=file)
        rewards = Evaluation(env=env, agent=agent, max_step=eval_step, episode=episode, seed=eval_seed)
        if saved_step <= train_step:
            data[saved_step//interval-1].extend(rewards) 
        env.close()
    return data
        

if __name__ == '__main__':
    p = Pool(len(train_seeds))
    data_list = p.map(worker, train_seeds)
    p.close()
    
    
    ### Visuallize ###
    data_list = np.array(data_list)
    len_s, len_d, len_e = data_list.shape
    data_list = data_list.transpose(1,0,2).reshape(len_d, len_s*len_e)
    
    LinePlot(data_list ,saved_steps, env_name, "QTableAgent", 'out')
    
