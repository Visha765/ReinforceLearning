from multiprocessing import Pool
import numpy as np
import sys, os
import time, datetime
import gym

from lib.training.Train import Train
from lib.training.Evaluation import Evaluation
from lib.util.fetchPikle import fetch_pikles
from lib.util.linePlot import LinePlot
from lib.model.replayQTableAgent import ReplayQTableAgent


### Condition ###
env_name = 'Pendulum-v0'

train_seeds = [11, 13, 17, 19, 23]    
train_step = 10000 #500000 
interval = 1000
K, L = 20, 18
buffer_size = train_step
batch_size = 256

eval_seed = 0
episode = 10 # 10
eval_step = 10000



def thread(train_seed):
    ### Train ###
    print('-'*10, "start Train", '-'*10)
    
    env = gym.make(env_name)
    env.seed(train_seed)
    np.random.seed(train_seed)
    agent = ReplayQTableAgent(K, L, buffer_size, batch_size)

    path=f"out/{agent.__class__.__name__ }_seed{train_seed}"
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(path):
        os.mkdir(path)
    
    start = time.time()
    Train(env=env, agent=agent, end_step=train_step, seed=train_seed, save_interval=interval, path=path)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(datetime.timedelta(seconds=elapsed_time)))
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", '-'*10)
    
    saved_steps, files = fetch_pikles(path)
    data_list = []
    for file in files:
        env = gym.make(env_name)
        env.seed(eval_seed)
        agent, _ = agent.load_models(path=path, filename=file)
        
        rewards = Evaluation(env=env, agent=agent, max_step=eval_step, episode=episode, seed=eval_seed)
        
        data_list.append(rewards)
        print(np.mean(rewards))
        env.close()
        
        
    ### Visualize ###
    LinePlot(data_list=data_list, label_list=saved_steps, env_name=env_name, seed=train_seed, path=path)


def main():
    p = Pool(len(train_seeds))
    p.map(thread, train_seeds)
    p.close()

if __name__ == '__main__':
    main()
    
