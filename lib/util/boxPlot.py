import numpy as np
import matplotlib.pyplot as plt
import os

def BoxPlot(data_list ,label_list, filename, path):
  fig, ax = plt.subplots()
  ax.boxplot(data_list)
  ax.set_xticklabels(label_list)
  ax.set_xlabel("step")
  ax.set_ylabel("reward")
  ax.set_title(filename)
  # plt.show()
  plt.savefig(os.path.join(path,filename)+'.png', format="png")
  
  
  
