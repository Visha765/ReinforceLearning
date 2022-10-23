from tqdm import tqdm
import gym
from compornent.Agent import *

env = gym.make("Pendulum-v0")
agent = Agent()
state = env.reset()

print(state, type(state))
print("min", env.observation_space.low)

for t in tqdm(range(1000)):
    env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
        state = env.reset()
env.close()