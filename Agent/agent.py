import numpy as np
import random

class Agent:
   
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros(state_size + (action_size,))
        self.learning_rate = learning_rate
        # Augmenter gamma à 0.99 (ou plus) pour se concentrer sur le long terme
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state + (best_next_action,)] * (not done)
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.learning_rate * td_error
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)