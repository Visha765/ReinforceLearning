from multiprocessing import Pool
from dataclasses import dataclass
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.worker import Worker
from lib.util.visuallize import Visuallize
from lib.model.qTableAgent import QTableAgent

### Condition ###
@dataclass
class args:
    env_name = "Pendulum-v0" # 環境名
    agent_name = "QTableAgent" # エージェント名      
    train_step = 500000 # 学習最大ステップ
    train_seed = 0 # 学習環境のseed値
    interval = 10000 # 状態を保存する間隔
    episode = 10 # 評価のエピソード数
    eval_step = 10000 # 評価最大ステップ数
    eval_seed = 0 # 評価環境のseed値
    K, L = 10, 9 # 状態、行動空間の各次元の分割数
    gamma=0.99 # 減衰率
    alpha=3*1e-4 # 学習率
    epsilon=0.05 # 行動方策のパラメータ
    
    def __init__(self, train_seed):
        self.train_seed = train_seed
    # return new agent
    def agent(self):
        return QTableAgent(self.K, self.L, \
                        self.gamma, self.alpha, self.epsilon)
    
train_seeds = [11, 13, 17, 19, 23]
args_list = [args(train_seed) for train_seed in train_seeds]

if __name__ == '__main__':
    ### Training & Evaluation ###
    p = Pool(len(args_list))
    data_list = p.map(Worker, args_list)
    p.close()
    
    ### Visuallize ###
    Visuallize(data_list, args_list[0])
