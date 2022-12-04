from multiprocessing import Pool
from dataclasses import dataclass
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.worker import Worker
from lib.model.qTableAgent import QTableAgent
from lib.model.replayQTableAgent import ReplayQTableAgent
from lib.util.transform import Transform
from lib.util.linePlot import LinePlot


### Condition ###
@dataclass
class params:
    env_name = "Pendulum-v0" # 環境名
    agent_name = "ReplayQTableAgent" # エージェント名
    dir_name = None # 保存先ディレクトリ
    train_step = 500000 # 学習最大ステップ
    train_seed = None # 学習環境のseed値
    interval = 5000 # 状態を保存する間隔
    episode = 10 # 評価のエピソード数
    eval_step = 10000 # 評価最大ステップ数
    eval_seed = 0 # 評価環境のseed値
    K, L = 10, 9 # 状態、行動空間の各次元の分割数
    gamma=0.99 # 減衰率
    alpha=3*1e-4 # 学習率
    epsilon=0.05 # 行動方策のパラメータ
    
    buffer_size = None # バッファーサイズ
    batch_size = 256 # バッチサイズ
    
    def __init__(self, train_seed, buffer_size):
        self.train_seed = train_seed
        self.buffer_size = buffer_size
        self.dir_name = f"{self.env_name}_{self.buffer_size}_{self.train_seed}" # 保存先ディレクトリ

    def agent(self):
        return ReplayQTableAgent(self.K, self.L, self.buffer_size, self.batch_size, \
                            self.gamma, self.alpha, self.epsilon)

train_seeds = [11, 13, 17, 19, 23]
buffer_sizes = [500000, 50000, 5000]
cond_list = []
for buffer_size in buffer_sizes:
    cond = [params(train_seed, buffer_size) for train_seed in train_seeds]
    cond_list.append(cond)
        

if __name__ == '__main__':
    data_list = []
    for cond in cond_list:
        p = Pool(len(train_seeds))
        data = p.map(Worker, cond)
        p.close()
        data = Transform(data)
        data_list.append(data)

    ### Visuallize ###
    label_list = [str(i) for i in buffer_sizes]
    saved_steps = [i for i in range(0, params.train_step+1, params.interval)]
    filename = "compare_buffer_size"
    LinePlot(data_list=data_list, label_list=label_list, x=saved_steps,
            filename=filename ,path="out")

