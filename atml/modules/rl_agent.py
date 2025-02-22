import numpy as np
import random

class RLAgent:
    def __init__(self, n_bins=10, actions=[-1, 0, 1], learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.n_bins = n_bins
        self.actions = actions
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.Q = np.zeros((n_bins, n_bins, len(actions)))
    
    def discretize(self, value, min_val, max_val):
        value = np.clip(value, min_val, max_val)
        return int(np.floor((value - min_val) / (max_val - min_val + 1e-10) * self.n_bins))
    
    def get_state(self, state_vector):
        rsi = state_vector[0] if not np.isnan(state_vector[0]) else 50
        macd_hist = state_vector[1] if not np.isnan(state_vector[1]) else 0
        rsi_bin = self.discretize(rsi, 0, 100)
        macd_bin = self.discretize(macd_hist, -5, 5)
        return (rsi_bin, macd_bin)
    
    def choose_action(self, state_vector):
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        s = self.get_state(state_vector)
        return self.actions[np.argmax(self.Q[s[0], s[1], :])]
    
    def update(self, state_vector, action, reward, next_state_vector):
        s = self.get_state(state_vector)
        s_next = self.get_state(next_state_vector)
        a_idx = self.actions.index(action)
        best_next = np.max(self.Q[s_next[0], s_next[1], :])
        self.Q[s[0], s[1], a_idx] += self.lr * (reward + self.discount * best_next - self.Q[s[0], s[1], a_idx])
