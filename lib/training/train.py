from tqdm import tqdm

def Train(env, agent, end_step, interval, path):
  agent.save_models(current_step=0, path=path)
  state = env.reset()
  for step in tqdm(range(end_step)):
    # env.render()
    action = agent.select_exploratory_action(state, step)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done, step)
    state = next_state
    if done:
      state = env.reset()
        
    ## save model per interval
    if ((step+1) % interval == 0):
      agent.save_models(current_step=step+1, path=path)
      agent.plot_loss(path=path)
      