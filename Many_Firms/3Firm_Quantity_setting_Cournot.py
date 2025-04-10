import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



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

def get_profits_three_firms(q1, q2, q3, a, b, c1, c2, c3):
    """Calculate profits for three firms."""
    P = price_function(a, b, q1 + q2 + q3)
    profit1 = P * q1 - c1 * q1
    profit2 = P * q2 - c2 * q2
    profit3 = P * q3 - c3 * q3
    return profit1, profit2, profit3

def plot_quantities(q1_hist, q2_hist, title="Cournot Learning Dynamics", smooth=False):
    plt.figure(figsize=(12, 6))
    
    if smooth:
        # Moving average for smoothing
        window = 500
        q1_smooth = np.convolve(q1_hist, np.ones(window)/window, mode='valid')
        q2_smooth = np.convolve(q2_hist, np.ones(window)/window, mode='valid')
        plt.plot(q1_smooth, label='Firm 1 (smoothed)')
        plt.plot(q2_smooth, label='Firm 2 (smoothed)')
    else:
        plt.plot(q1_hist, label='Firm 1')
        plt.plot(q2_hist, label='Firm 2')

    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_cournot_simulation(num_episodes, max_q, delta_n, a, b, c1, c2,
                           alpha, epsilon, epsilon_decay, min_epsilon,
                           initial_q1=None, initial_q2=None):
    
    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)

    # Allow user to specify initial quantities
    if initial_q1 is None:
        q1 = max_q // 2
    else:
        q1 = initial_q1

    if initial_q2 is None:
        q2 = max_q // 2
    else:
        q2 = initial_q2

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

        q1, q2 = q1_new, q2_new

    return agent1.history, agent2.history

def run_cournot_simulation_three_firms(num_episodes, max_q, delta_n, a, b, c1, c2, c3,
                                       alpha, epsilon, epsilon_decay, min_epsilon,
                                       initial_q1=None, initial_q2=None, initial_q3=None):
    """Run Cournot simulation for three firms."""
    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent3 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)

    # Allow user to specify initial quantities
    q1 = initial_q1 if initial_q1 is not None else max_q // 3
    q2 = initial_q2 if initial_q2 is not None else max_q // 3
    q3 = initial_q3 if initial_q3 is not None else max_q // 3

    for episode in range(num_episodes):
        state1 = (q1, q2, q3)
        state2 = (q2, q3, q1)
        state3 = (q3, q1, q2)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)
        action3 = agent3.choose_action(state3)

        q1_new = q1 + action1
        q2_new = q2 + action2
        q3_new = q3 + action3

        reward1, reward2, reward3 = get_profits_three_firms(q1_new, q2_new, q3_new, a, b, c1, c2, c3)

        next_state1 = (q1_new, q2_new, q3_new)
        next_state2 = (q2_new, q3_new, q1_new)
        next_state3 = (q3_new, q1_new, q2_new)

        agent1.update(state1, action1, reward1, next_state1)
        agent2.update(state2, action2, reward2, next_state2)
        agent3.update(state3, action3, reward3, next_state3)

        agent1.decay_epsilon()
        agent2.decay_epsilon()
        agent3.decay_epsilon()

        q1, q2, q3 = q1_new, q2_new, q3_new

    return agent1.history, agent2.history, agent3.history


# Example usage
if __name__ == "__main__":

    # Example usage for 3 firms
    q1_hist, q2_hist, q3_hist = run_cournot_simulation_three_firms(
        num_episodes=1500000,
        max_q=20,  # Maximum quantity a firm can produce
        delta_n=2,  # Change in production allowed
        a=-1,  # Coefficient for price function (negative slope)
        b=40,  # Intercept for price function (base price when total quantity is 0)
        c1=6,  # Cost per unit for Firm 1
        c2=6,  # Cost per unit for Firm 2
        c3=6,  # Cost per unit for Firm 3
        alpha=0.1,  # Learning rate
        epsilon=1.0,  # Initial exploration rate
        epsilon_decay=0.995,  # Decay rate for exploration
        min_epsilon=0.05,  # Minimum exploration rate
        initial_q1=0,  # Initial quantity for Firm 1
        initial_q2=0,  # Initial quantity for Firm 2
        initial_q3=0   # Initial quantity for Firm 3
    )

    # Plot results for 3 firms with smoothing
    plt.figure(figsize=(12, 6))

    # Define the smoothing window size
    window_size = 10000

    # Apply moving average for smoothing
    q1_smooth = np.convolve(q1_hist, np.ones(window_size)/window_size, mode='valid')
    q2_smooth = np.convolve(q2_hist, np.ones(window_size)/window_size, mode='valid')
    q3_smooth = np.convolve(q3_hist, np.ones(window_size)/window_size, mode='valid')

    # Plot the smoothed quantities
    plt.plot(q1_smooth, label="Firm 1 (smoothed)", color="blue")
    plt.plot(q2_smooth, label="Firm 2 (smoothed)", color="orange")
    plt.plot(q3_smooth, label="Firm 3 (smoothed)", color="green")

    # Add labels, title, and legend
    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title("Smoothed Cournot Learning Dynamics for 3 Firms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

