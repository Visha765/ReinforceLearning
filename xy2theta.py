import numpy as np

def xy2theta(state):
  cos, sin = state[0:2]
  theta = np.arccos(cos) if sin > 0 else -np.arccos(cos)
  return (theta, state[2])