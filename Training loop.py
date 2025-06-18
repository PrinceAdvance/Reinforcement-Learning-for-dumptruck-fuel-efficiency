env = MineEnv(size=5)
agent = QLearningAgent(env)

episodes = 500

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    for step in range(50):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if done:
            break
    if ep % 50 == 0:
        print(f"Episode {ep} reward: {total_reward}")
