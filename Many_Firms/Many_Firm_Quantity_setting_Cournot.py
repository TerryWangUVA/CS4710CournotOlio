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

def get_profits_multiple_firms(quantities, a, b, costs):
    """Calculate profits for multiple firms."""
    q_total = sum(quantities)
    P = price_function(a, b, q_total)
    profits = [P * q - c * q for q, c in zip(quantities, costs)]
    return profits

def run_cournot_simulation_multiple_firms(num_episodes, num_firms, max_q, delta_n, a, b, costs,
                                          alpha, epsilon, epsilon_decay, min_epsilon, initial_quantities=None):
    """Run Cournot simulation for multiple firms."""
    agents = [CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon) for _ in range(num_firms)]

    # Initialize quantities
    if initial_quantities is None:
        quantities = [max_q // num_firms] * num_firms
    else:
        quantities = initial_quantities

    histories = [[] for _ in range(num_firms)]

    count = 0
    for episode in range(num_episodes):
        # Calculate the percentage of progress
        percentage = ((episode + 1) * 100)  //  num_episodes

        # Print percentage only when it changes (integer)
        if percentage > count:
            count = percentage
            print(count, end=" ", flush=True)

        states = [tuple(quantities)] * num_firms
        actions = [agent.choose_action(state) for agent, state in zip(agents, states)]

        # Update quantities based on actions
        new_quantities = [max(0, min(max_q, q + a)) for q, a in zip(quantities, actions)]

        # Calculate rewards (profits)
        rewards = get_profits_multiple_firms(new_quantities, a, b, costs)

        # Update agents
        next_states = [tuple(new_quantities)] * num_firms
        for i, agent in enumerate(agents):
            agent.update(states[i], actions[i], rewards[i], next_states[i])
            agent.decay_epsilon()

        # Update quantities and histories
        quantities = new_quantities
        for i in range(num_firms):
            histories[i].append(quantities[i])

    return histories


# Example usage
if __name__ == "__main__":

    # Example usage for x firms
    num_firms = 3
    costs = [3] * num_firms  # Cost per unit for each firm
    initial_quantities = [0] * num_firms  # Initial quantities for each firm

    histories = run_cournot_simulation_multiple_firms(
        num_episodes=700000,
        num_firms=num_firms,
        max_q=24,  # Maximum quantity a firm can produce
        delta_n=1,  # Change in production allowed
        a=-1,  # Coefficient for price function (negative slope)
        b=50,  # Intercept for price function (base price when total quantity is 0)
        costs=costs,
        alpha=0.1,  # Learning rate
        epsilon=1.0,  # Initial exploration rate
        epsilon_decay=0.995,  # Decay rate for exploration
        min_epsilon=0.05,  # Minimum exploration rate
        initial_quantities=initial_quantities
    )

    # Plot results for many firms with smoothing
    plt.figure(figsize=(12, 7))

    # Define the smoothing window size
    window_size = 5000

    # Apply moving average for smoothing and plot each firm's quantities
    for i, history in enumerate(histories):
        smooth_history = np.convolve(history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smooth_history, label=f"Firm {i + 1} (smoothed)", linewidth=.5)  # Set linewidth to 1 for thinner lines
    # Add labels, title, and legend
    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title("Smoothed Cournot Learning Dynamics for many Firms")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

