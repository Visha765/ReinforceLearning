from multiprocessing import Pool
from dataclasses import dataclass
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.worker import Worker
from lib.util.visuallize import Visuallize
from lib.model.replayQTableAgent import ReplayQTableAgent


### Condition ###
@dataclass
class args:
    env_name = "Pendulum-v0"
    agent_name = "ReplayQTableAgent"       
    train_step = 500000
    train_seed = 0
    interval = 10000
    episode = 10 # 10
    eval_step = 10000 #maxstep
    eval_seed = 0
    K, L = 10, 9
    buffer_size = train_step
    batch_size = 256
    
    # changed in each process
    def __init__(self, train_seed):
        self.train_seed = train_seed
    # return new agent
    def agent(self):
        return ReplayQTableAgent(self.K, self.L, self.buffer_size, self.batch_size)
    
train_seeds = [11, 13, 17, 19, 23]
args_list = [args(train_seed) for train_seed in train_seeds]

if __name__ == '__main__':
    ### Training & Evaluation ###
    p = Pool(len(train_seeds))
    data_list = p.map(Worker, args_list)
    p.close()
    
    ### Visuallize ###
    Visuallize(data_list, args_list[0], train_seeds)
