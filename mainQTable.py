import numpy as np
import sys, os
import gym

from lib.training.Train import Train
from lib.training.Evaluation import Evaluation
from lib.util.fetchPikle import fetch_pikles
from lib.util.linePlot import LinePlot
from lib.model.qTableAgent import QTableAgent

def main():

    env_name = 'Pendulum-v0'
        
    train_step = 100000 # 1000
    train_seed = 8243
    interval = 1000
    K, L = 10, 9
    
    episode = 10 # 10
    eval_step = 1000
    eval_seed = 41
    
    
    path=f"out/{env_name}_seed{train_seed}"
    if not os.path.exists(path):
        os.mkdir(path)
        
        
    ### Train ###
    print('-'*10, "start Train", '-'*10)
    
    env = gym.make(env_name)
    env.seed(train_seed)
    env.action_space.seed(train_seed)
    np.random.seed(train_seed)
    agent = QTableAgent(K, L)
    start_step = 0
    
    ## load model if exists ##
    saved_steps, files = fetch_pikles(path)
    if len(saved_steps)!=0: # none savedata
        env, agent, start_step = agent.load_models(path, files[-1]) # load latest savedata 
    
    Train(env=env, agent=agent, start_step=start_step, end_step=train_step, seed=train_seed, save_interval=interval, path=path)
    env.close()
    
    
    ### Evaluation ###
    print('-'*10, "start Evaluation", '-'*10)
    
    saved_steps, files = fetch_pikles(path)
    data_list = []
    for file in files:
        env = gym.make(env_name)
        env.seed(eval_seed)
        # agent = QTableAgent(K, L)
        _, agent, _ = agent.load_models(path=path, filename=file)
        
        rewards = Evaluation(env=env, agent=agent, max_step=eval_step, episode=episode, seed=eval_seed)
        
        data_list.append(rewards)
        print(np.mean(rewards))
        
        
    ### Visualize ###
    LinePlot(data_list=data_list, label_list=saved_steps, env_name=env_name, seed=train_seed, path=path)

if __name__ == '__main__':
    main()
    
