from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os

def LinePlot(data_list ,label_list, env_name, seed, path):
  plt.rcParams['font.family'] = 'Hiragino Sans'
  
  f = [np.percentile(data, q=[25,50, 75]) for data in data_list]
  f = np.array(f).T

  plt.plot(label_list, f[2], label="75%")
  plt.plot(label_list, f[1], label="50%", color="blue", marker='x')
  plt.plot(label_list, f[0], label="25%")
  
  plt.fill_between(label_list, f[0], f[2], color="green", alpha=0.3)
  
  title = f"{env_name}_seed{seed}_step{label_list[-1]}"
  plt.title(title, fontsize = 16)
  plt.xlabel('学習ステップ', fontsize = 12)
  plt.ylabel('累積報酬', fontsize = 12)
  # plt.tick_params(labelsize=8)
  plt.grid(True)
  plt.legend(loc="upper right", fontsize=12)
  
  plt.savefig(os.path.join(path,title)+'.png', format="png")
  plt.show()
