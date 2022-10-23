import gym
import numpy as np

env = gym.make('Pendulum-v0')
state = env.reset()
print(env.observation_space)
for t in range(0): 
  env.render()
  action = env.action_space.sample()
  state, reward, done, info = env.step(action) 
  print(state[0]**2 + state[1]**2)
  print("ob_state",state)
  print("env_state",env.env.state)
  print(np.cos(env.env.state[0]))
  print(action)
  if done:
    break
env.close()