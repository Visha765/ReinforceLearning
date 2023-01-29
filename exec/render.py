import gym
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.model.TD3_agent import TD3Agent

episode = 1000
seed = sys.argv[1]
step = sys.argv[2]

env = gym.make("Pendulum-v0")
agent = TD3Agent(0,0)
agent.load_models(saved_step=step, path=f"out/Pendulum-v0_TD3_{seed}")

for it in range(episode):
  state = env.reset()
  env.render()
  action = agent.select_action(state)
  next_state, reward, done, info = env.step(action)
  state = next_state
  if done:
    break 

env.close()