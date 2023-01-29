import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list2tensor(x):
  x = np.array(x, dtype=np.float32)
  return torch.Tensor(x).to(device)