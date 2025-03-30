import numpy as np
import random

class CournotAgent:
    def __init__(self, max_quantity, delta_n, alpha, epsilon, epsilon_decay, min_epsilon):
        self.max_q = max_quantity
        self.delta_n = delta_n
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.action_space = list(range(-delta_n, delta_n + 1))  # Change in production
        self.q_table = {}  # Use dictionary due to large state space
        self.history = []

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        legal_actions = self.get_legal_actions(state)
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            q_values = [self.get_q(state, a) for a in legal_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q]
            action = random.choice(best_actions)
        self.history.append(state[0] + action)
        return action

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in self.get_legal_actions(next_state)])
        new_q = old_q + self.alpha * (reward - old_q)  # No discounting
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_legal_actions(self, state):
        current_q = state[0]
        return [a for a in self.action_space if 0 <= current_q + a <= self.max_q]

# Economic parameters
def price_function(a, b, q_total):
    return a * q_total + b

def get_profits(q1, q2, a, b, c1, c2):
    P = price_function(a, b, q1 + q2)
    profit1 = P * q1 - c1 * q1
    profit2 = P * q2 - c2 * q2
    return profit1, profit2

def run_cournot_simulation(num_episodes, max_q, delta_n, a, b, c1, c2,
                           alpha, epsilon, epsilon_decay, min_epsilon):
    
    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)

    # Initial quantities
    q1, q2 = max_q // 2, max_q // 2

    for episode in range(num_episodes):
        state1 = (q1, q2)
        state2 = (q2, q1)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)

        q1_new = q1 + action1
        q2_new = q2 + action2

        reward1, reward2 = get_profits(q1_new, q2_new, a, b, c1, c2)

        next_state1 = (q1_new, q2_new)
        next_state2 = (q2_new, q1_new)

        agent1.update(state1, action1, reward1, next_state1)
        agent2.update(state2, action2, reward2, next_state2)

        agent1.decay_epsilon()
        agent2.decay_epsilon()

        q1, q2 = q1_new, q2_new  # Update for next episode

    return agent1.history, agent2.history

# Example usage
if __name__ == "__main__":
    q1_hist, q2_hist = run_cournot_simulation(
        num_episodes=1000000,
        max_q=100,
        delta_n=2,
        a=-1,
        b=100,
        c1=20,
        c2=20,
        alpha=0.1,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05
    )
    #analytical result for the set of parameters given above is q1=q2=26.67.

    #print("Firm 1 quantity history:", q1_hist[:10])
    #print("Firm 2 quantity history:", q2_hist[:10])
    print("Last 10 rounds of learning results:")
    print("Firm 1 quantities:", q1_hist[-10:])
    print("Firm 2 quantities:", q2_hist[-10:])