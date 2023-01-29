import numpy as np
import matplotlib.pyplot as plt
import os

def LossPlot(data_list, x, filename, path):
  plt.figure()
  plt.rcParams['font.family'] = 'Noto Sans JP', 'Hiragino Sans'
  
  plt.plot(x, data_list, marker='.')
  plt.title(filename, fontsize = 16)
  plt.xlabel('学習ステップ', fontsize = 12)
  plt.ylabel('Loss', fontsize = 12)
  plt.grid(True)
  
  plt.savefig(os.path.join(path, filename)+'.png', format="png")
  # plt.show()
  plt.close()
