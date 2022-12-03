def Evaluate(env, agent, max_step, episode):
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
