class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.env = env

    def get_q(self, state, action):
        """Return Q-value for (state, action), default 0"""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_vals = [self.get_q(state, a) for a in ACTIONS]
        max_q = max(q_vals)
        return random.choice([a for a, q in zip(ACTIONS, q_vals) if q == max_q])

    def update(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation"""
        old_q = self.get_q(state, action)
        future_q = max([self.get_q(next_state, a) for a in ACTIONS])
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table[(state, action)] = new_q
