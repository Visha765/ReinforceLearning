from multiprocessing import Pool
import torch.multiprocessing as multiprocessing
from dataclasses import dataclass
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.worker import Worker
from lib.util.transform import Transform
from lib.util.line_plot import LinePlot
from lib.model.TD3_agent_bi import TD3Agent

if multiprocessing.get_start_method() == 'fork':
    multiprocessing.set_start_method('spawn', force=True)
    print("{} setup done".format(multiprocessing.get_start_method()))

### Condition ###
@dataclass
class params:
    env_name = "BipedalWalker-v3" # 環境名
    agent_name = "TD3" # エージェント名
    dir_name = None # 保存先ディレクトリ
    train_step = 100000 # 学習最大ステップ
    train_seed = None # 学習環境のseed値
    interval = 1000 # 状態を保存する間隔
    episode = 30 # 評価のエピソード数
    eval_step = 10000 # 評価最大ステップ数
    eval_seed = 0 # 評価環境のseed値

    buffer_size = train_step
    batch_size = 256
    
    tau = 0.03  # target networkの更新率
    
    def __init__(self, train_seed):
        self.train_seed = train_seed
        self.dir_name = f"{self.env_name}_{self.agent_name}_{self.train_seed}" # 保存先ディレクトリ
        
    def agent(self, env):
        return TD3Agent(env, self.buffer_size, self.batch_size, target_tau=self.tau)
    
# train_seeds = [100, 200, 300, 400, 500]
train_seeds = [11]
condition = [params(train_seed) for train_seed in train_seeds]

if __name__ == '__main__':
    data_list = []
    p = Pool(len(condition))
    data = p.map(Worker, condition)
    p.close()
    data = Transform(data)
    data_list.append(data)

    ### Visuallize ###
    label_list = ["TD3"]
    saved_steps = [i for i in range(0, params.train_step+1, params.interval)]
    filename = "TD3"
    LinePlot(data_list=data_list, label_list=label_list, x=saved_steps,
            filename=filename, path="out")

