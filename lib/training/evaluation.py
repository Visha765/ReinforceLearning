def Evaluation(env, agent, max_step, episode, seed):

  rewards = []
  for it in range(episode):
    state = env.reset()
    sum_reward = 0
    for t in range(max_step):
      # env.render()
      action = agent.select_action(state)
      next_state, reward, done, info = env.step(action)
      sum_reward += reward
      state = next_state
      if done:
        break 
    rewards.append(sum_reward)

  return rewards;
