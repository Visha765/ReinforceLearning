from multiprocessing import Pool
from dataclasses import dataclass
import numpy as np
import sys, os
import glob
import gym

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.train import Train
from lib.training.evaluation import Evaluation
from lib.util.linePlot import LinePlot
from lib.model.qTableAgent import QTableAgent

@dataclass
class args:
    env_name= "Pendulum-v0"
    agent_name= "QTableAgent"       
    train_step = 500000
    interval = 10000
    episode = 10 # 10
    eval_step = 10000 #maxstep
    eval_seed = 0
    K, L = 10, 9
    
    def agent(self):
        QTableAgent(self.K, self.L)

train_seeds = [11, 13, 17, 19, 23]
d = args()


def worker(train_seed, d):
    ## Train ###
    print('-'*10, "start Train", '-'*10)
    
    env = gym.make(d.env_name)
    env.seed(train_seed)
    np.random.seed(train_seed)
    agent = d.agent()

    path = f"out/{d.env_name}_{d.agent_name}_seed{train_seed}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)
    
    Train(env=env, agent=agent, end_step=d.train_step, seed=train_seed, save_interval=d.interval, path=path)
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", '-'*10)
    
    files = glob.glob(os.path.join(path, "*.pickle"))
    files = [os.path.split(file)[1] for file in files]
    data = [[] for i in range(d.train_step//d.interval)]
    for file in files:
        env = gym.make(d.env_name)
        env.seed(d.eval_seed)
        agent, saved_step= agent.load_models(path=path, filename=file)
        rewards = Evaluation(env=env, agent=agent, max_step=d.eval_step, episode=d.episode, seed=d.eval_seed)
        if saved_step <= d.train_step:
            data[saved_step//d.interval-1].extend(rewards) 
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
    saved_steps = [i for i in range(d.interval, d.train_step+1, d.interval)]
    
    LinePlot(data_list ,saved_steps, f"{d.env_name}_{d.agent_name}", 'out')
    
    import pickle
    with open(os.path.join('out', f'{d.env_name}_{d.agent_name}_{train_seeds}.pickel'), 'wb') as f:
        data = (data_list, saved_steps)
        pickle.dump(data, f)
