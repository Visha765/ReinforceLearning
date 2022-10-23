import numpy as np
import matplotlib.pyplot as plt
import os

def BoxPlot(data_list ,label_list, env_name, seed, path):
  title = f"{env_name}_seed{seed}_step[0_{label_list[-1]}]"
  fig, ax = plt.subplots()
  ax.boxplot(data_list)
  ax.set_xticklabels(label_list)
  ax.set_xlabel("step")
  ax.set_ylabel("reward")
  ax.set_title(title)
  # plt.show()
  plt.savefig(os.path.join(path,title)+'.png', format="png")
  
  
  
