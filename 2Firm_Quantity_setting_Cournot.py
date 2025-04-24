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


# Example usage
if __name__ == "__main__":
    q1_hist, q2_hist = run_cournot_simulation(
        num_episodes=300000,
        max_q=20, # Maximum quantity a firm can produce
        delta_n=2, # Change in production allowed
        a=-1, # Coefficient for price function (negative slope)
        b=40, # Intercept for price function (base price when total quantity is 0)
        c1=4, # Cost per unit for Firm 1
        c2=4, # Cost per unit for Firm 2
        alpha=0.1, # Learning rate
        epsilon=1.0, # Initial exploration rate
        epsilon_decay=0.995, # Decay rate for exploration
        min_epsilon=0.05, # Minimum exploration rate
        initial_q1=0, # Initial quantity for Firm 1
        initial_q2=0 # Initial quantity for Firm 2
    )
    #analytical result for the set of parameters given above is q1=q2=12.
    # Formula : (b-c)/a(2+1) = 12
    # Additionally: price=16 profits= 144 respectively.

    print("Test Case 1:")
    print("Firm 1 quantity history 1-10:", q1_hist[:10])
    print("Firm 2 quantity history 1-10:", q2_hist[:10])
    print("Last 10 rounds of learning results:")
    print("Firm 1 quantities in last 10 episodes:", q1_hist[-10:])
    print("Firm 2 quantities in last 10 episodes:", q2_hist[-10:])
    print("Firm 1 average quantity in the last 10 episodes:", np.mean(q1_hist[-10:]))
    print("Firm 2 average quantity in the last 10 episodes:", np.mean(q2_hist[-10:]))

    q12_hist, q22_hist = run_cournot_simulation(
        num_episodes=300000,
        max_q=100,
        delta_n=2,
        a=-1,
        b=100,
        c1=10,
        c2=90,
        alpha=0.1,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.001,
        initial_q1=15,
        initial_q2=15
    )
    #A little sanity check for corner cases.
    # just to verify that the model works for corner solutions as well
    # in this case q2*=0, and q1*=45
    # sometimes this case fails, but I am not sure why. It is VERY, VERY UNLIKELY to fail. 
    print("Test Case 2:")
    print("Firm 1 quantity history 1-10:", q12_hist[:10])
    print("Firm 2 quantity history 1-10:", q22_hist[:10])
    print("Last 10 rounds of learning results:")
    print("Firm 1 quantities in last 10 episodes:", q12_hist[-10:])
    print("Firm 2 quantities in last 10 episodes:", q22_hist[-10:])
    print("Firm 1 average quantity in the last 10 episodes:", np.mean(q12_hist[-10:]))
    print("Firm 2 average quantity in the last 10 episodes:", np.mean(q22_hist[-10:]))

    #An attempt to visualize outputs for case 1:
    plot_quantities(q1_hist, q2_hist, title="Test Case 1: Symmetric Cournot (q1 = q2 ≈ 12)", smooth=True)
    plot_quantities(q12_hist, q22_hist, title="Test Case 2: Corner Solution (q1=45, q2=0))", smooth=True)

    # Calculate the difference in quantities (Firm 1 - Firm 2)
    quantity_differences = [q1 - q2 for q1, q2 in zip(q1_hist, q2_hist)]

    # Smooth the quantity differences using a moving average
    window_size = 6000  # Adjust the window size for smoothing
    smoothed_quantity_differences = np.convolve(quantity_differences, np.ones(window_size)/window_size, mode='valid')

    # Plot the smoothed difference in quantities
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_quantity_differences, label="Smoothed Quantity Difference (Firm 1 - Firm 2)", color="blue")
    plt.axhline(0, color="red", linestyle="--", label="Convergence Line (0)")
    plt.xlabel("Episode")
    plt.ylabel("Quantity Difference (Firm 1 - Firm 2)")
    plt.title("Convergence of Quantities Produced by Firms Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

