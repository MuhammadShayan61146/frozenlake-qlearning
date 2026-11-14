"""
agent.py

Implements the QLearningAgent: holds the Q-table, chooses actions via
epsilon-greedy exploration, and updates the Q-table using the Q-Learning
(off-policy) update rule.
"""

import numpy as np


class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha, gamma,
                 epsilon_start, epsilon_min, epsilon_decay, rng):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng  # a numpy random Generator, for reproducible randomness

        # Q-table: one row per state, one column per action, start at 0.
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))  # explore
        return int(np.argmax(self.q_table[state]))            # exploit

    def update(self, state, action, reward, next_state, terminated):
        """
        Q-Learning update rule:
            Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        where target = reward + gamma * max_a' Q(s', a')

        IMPORTANT: if the episode terminated (goal reached or fell in a
        hole), there is no valid "next action" to bootstrap from, so the
        future-value term must be zero. Otherwise we would incorrectly
        add value from beyond the end of the episode.
        """
        best_next_value = 0.0 if terminated else np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next_value
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)  # exposed so callers can track convergence

    def decay_epsilon(self):
        """Called once per episode to shift from exploration to exploitation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
