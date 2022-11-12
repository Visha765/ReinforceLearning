import numpy as np

class QTable():
  
  def __init__(self, K :int ,L :int):
    self.gamma = 0.99
    self.alpha = 3*1e-4

    self.K = K
    self.L = L
    
    theta = (-np.pi, np.pi)
    omega = (-8, 8)
    tau = (-2, 2)
    
    self.bin_theta = np.linspace(*theta, self.K+1)[1:-1]
    self.bin_omega = np.linspace(*omega, self.K+1)[1:-1]
    self.bin_tau = np.linspace(*tau, self.L+1)[1:-1]
    
    a = np.linspace(-2, 2, self.L+1)
    self.actions = [(a[i]+a[i+1])/2 for i in range(self.L)]
    
    self.table= 1e-8 * np.random.normal(0, 1, size=(self.K*self.K, self.L))
    
  # state: (cos(theta), sin(theta), omega) --> (theta, omega)
  def xy2theta(self, state):
    cos, sin = state[0:2]
    theta = np.arccos(cos) if sin > 0 else -np.arccos(cos)
    return (theta, state[2])
  
  def digitize_state(self, state):
    theta, omega = self.xy2theta(state) 
    digitized = [
        np.digitize(theta, bins=self.bin_theta),
        np.digitize(omega, bins=self.bin_omega)
    ]
    return sum([x * (self.K**i) for i, x in enumerate(digitized)])
  
  def digitize_action(self, action):
    action = action[0]
    return np.digitize(action, bins=self.bin_tau)
  
  def get_maxQ_action(self, state):
    idx_state = self.digitize_state(state)
    idx_action = np.argmax(self.table[idx_state])
    return self.actions[idx_action]
  
  def update_qTable(self, state, action, reward, next_state, done=0):
    idx_state = self.digitize_state(state)
    idx_action = self.digitize_action(action)
    idx_next_state = self.digitize_state(next_state)
    next_max_Q = max(self.table[idx_next_state])

    self.table[idx_state, idx_action] = (1-self.alpha) * self.table[idx_state, idx_action] \
      + self.alpha * (reward + (1 - done) * self.gamma * next_max_Q)
    

