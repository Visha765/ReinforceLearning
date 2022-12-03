from tqdm import tqdm

def Train(env, agent, end_step, interval, path):
  agent.save_models(env=env, current_step=0, path=path)
  state = env.reset()
  for t in tqdm(range(end_step)):
    # env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
      state = env.reset()
        
    ## save model per interval
    if ((t+1) % interval == 0):
      agent.save_models(env=env, current_step=t+1, path=path)