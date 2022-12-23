from multiprocessing import Pool
from dataclasses import dataclass
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.worker import Worker
from lib.util.transform import Transform
from lib.util.line_plot import LinePlot
from lib.model.actor_critic_agent import ActorCriticAgent

### Condition ###
@dataclass
class params:
    env_name = "Pendulum-v0" # 環境名
    agent_name = "ActorCritic" # エージェント名
    dir_name = None # 保存先ディレクトリ
    train_step = 10000 # 学習最大ステップ
    train_seed = None # 学習環境のseed値
    interval = 1000 # 状態を保存する間隔
    episode = 10 # 評価のエピソード数
    eval_step = 5000 # 評価最大ステップ数
    eval_seed = 0 # 評価環境のseed値
    gamma = 0.99 # 減衰率

    buffer_size = train_step
    batch_size = 32
    
    def __init__(self, train_seed):
        self.train_seed = train_seed
        self.dir_name = f"{self.env_name}_{self.agent_name}_{self.train_seed}" # 保存先ディレクトリ
        
    def agent(self):
        return ActorCriticAgent(self.buffer_size, self.batch_size)
    
train_seeds = [11, 13, 17, 19, 23]
conditions = [params(train_seed) for train_seed in train_seeds]

if __name__ == '__main__':
    data_list = []
    p = Pool(len(conditions))
    data = p.map(Worker, conditions)
    p.close()
    data = Transform(data)
    data_list.append(data)

    ### Visuallize ###
    label_list = ["方法1"]
    saved_steps = [i for i in range(0, params.train_step+1, params.interval)]
    filename = "ActorCritic"
    LinePlot(data_list=data_list, label_list=label_list, x=saved_steps,
            filename=filename, path="out")

