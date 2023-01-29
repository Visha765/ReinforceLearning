from multiprocessing import Pool
import torch.multiprocessing as multiprocessing
from dataclasses import dataclass
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.training.worker import Worker
from lib.util.transform import Transform
from lib.util.line_plot import LinePlot
from lib.model.TD3_agent import TD3Agent
from lib.model.TD3_agent_ablation import TD3Agent_withoutTATC, TD3Agent_withoutTPSR, TD3Agent_withoutDPU, TD3Agent_withoutCDQ

if multiprocessing.get_start_method() == 'fork':
    multiprocessing.set_start_method('spawn', force=True)
    print("{} setup done".format(multiprocessing.get_start_method()))
    
### Condition ###
@dataclass
class params:
  env_name = "Pendulum-v0" # 環境名
  agent_name = None # エージェント名
  dir_name = None # 保存先ディレクトリ
  train_step = 100000 # 学習最大ステップ
  train_seed = None # 学習環境のseed値
  interval = 1000 # 状態を保存する間隔
  episode = 30 # 評価のエピソード数
  eval_step = 100000 # 評価最大ステップ数
  eval_seed = 0 # 評価環境のseed値

  buffer_size = train_step
  batch_size = 256
  
  target_tau = 0.03 # target networkの更新率
  
  def __init__(self, train_seed):
    self.train_seed = train_seed
    self.dir_name = f"{self.env_name}_{self.agent_name}_{self.train_seed}" # 保存先ディレクトリ
      
  def agent(self, env):
    pass
  
  
class params_TD3(params):
  agent_name = "TD3"
  def agent(self, env):
    return TD3Agent(env, self.buffer_size, self.batch_size, target_tau=self.target_tau)
  
options = ["TATC", "TPSR", "DPU", "CDQ"]
for option in options:
  exec(f"""
class params_TD3_without{option}(params):
  agent_name = "TD3-{option}"
  def agent(self, env):
    return TD3Agent_without{option}(env, self.buffer_size, self.batch_size, target_tau=self.target_tau)
  """)

train_seeds = [100 , 200, 300, 400, 500]

conditions = [[params_TD3(train_seed) for train_seed in train_seeds]]
for option in options:
  exec(f"""
conditions.append([params_TD3_without{option}(train_seed) for train_seed in train_seeds])
  """)
  
  
if __name__ == '__main__':
  data_list = []
  for condition in conditions:
    p = Pool(len(condition))
    data = p.map(Worker, condition)
    p.close()
    data = Transform(data)
    data_list.append(data)

  ### Visuallize ###
  label_list = ["origin"] + options
  saved_steps = [i for i in range(0, params.train_step+1, params.interval)]
  filename = "TD3-comp"
  LinePlot(data_list=data_list, label_list=label_list, x=saved_steps,
          filename=filename, path="out")

