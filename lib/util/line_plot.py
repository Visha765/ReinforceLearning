import numpy as np
import matplotlib.pyplot as plt
import os

def LinePlot(data_list, label_list , x, filename, path):
  plt.figure()
  plt.rcParams['font.family'] = 'Noto Sans JP', 'Hiragino Sans'
  color_list = ['blue', 'red', 'green', 'orange', 'purple']
  
  for data, label, color  in zip(data_list, label_list, color_list):
    f = [np.percentile(i, q=[25,50, 75]) for i in data]
    f = np.array(f).T
    plt.plot(x, f[1], label=label, color=color)
    plt.fill_between(x, f[0], f[2], color=color, alpha=0.25)
  
  plt.title(filename, fontsize = 16)
  plt.xlabel('学習ステップ', fontsize = 12)
  plt.ylabel('累積報酬', fontsize = 12)
  plt.grid(True)
  plt.legend(loc="lower right", fontsize=12)
  
  plt.savefig(os.path.join(path, filename)+'.png', format="png")
  # plt.show()
  plt.close()