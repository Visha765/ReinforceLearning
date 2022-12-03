import numpy as np

def Transform(data):
  data = np.array(data)
  len_s, len_d, len_e = data.shape
  data = data.transpose(1,0,2).reshape(len_d, len_s*len_e)
  return data