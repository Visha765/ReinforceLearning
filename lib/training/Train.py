from tqdm import tqdm
import gym

def Train(env, agent, start_step, end_step, seed, save_interval, path):
  state = env.reset()
  
  for t in tqdm(range(start_step, end_step)):
      
    # env.render()
    action = agent.select_exploratory_action(state)
    next_state, reward, done, info = env.step(action)
    agent.train(state, action, next_state, reward, done)
    state = next_state
    if done:
      state = env.reset()
        
    ## save model per interval
    if ((t+1) % save_interval == 0 and (t+1) != 0):
      agent.save_models(
                        env=env,
                        current_step=t+1, 
                        seed=seed, 
                        path=path
                      )

  return agent