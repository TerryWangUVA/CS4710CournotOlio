import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time  # Import the time module for elapsed time calculation


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


def get_profits_three_firms(q1, q2, q3, q4, a, b, c1, c2, c3, c4):
    """Calculate profits for three firms."""
    P = price_function(a, b, q1 + q2 + q3 + q4)
    profit1 = P * q1 - c1 * q1
    profit2 = P * q2 - c2 * q2
    profit3 = P * q3 - c3 * q3
    profit4 = P * q4 - c4 * q4
    
    return profit1, profit2, profit3, profit4

def run_cournot_simulation_three_firms(num_episodes, max_q, delta_n, a, b, c1, c2, c3, c4, 
                                       alpha, epsilon, epsilon_decay, min_epsilon,
                                       initial_q1=None, initial_q2=None, initial_q3=None, initial_q4=None):
    """Run Cournot simulation for three firms."""
    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent3 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent4 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    

    # Allow user to specify initial quantities
    q1 = initial_q1 if initial_q1 is not None else max_q // 4
    q2 = initial_q2 if initial_q2 is not None else max_q // 4
    q3 = initial_q3 if initial_q3 is not None else max_q // 4
    q4 = initial_q4 if initial_q4 is not None else max_q // 4
    
    count = 0
    for episode in range(num_episodes):
        # Calculate the percentage of progress
        percentage = ((episode + 1) * 100)  //  num_episodes

        # Print percentage only when it changes (integer)
        if percentage > count:
            count = percentage
            print(count, end=" ", flush=True)
            
        state1 = (q1, q2, q3, q4)
        state2 = (q2, q3, q4, q1)
        state3 = (q3, q4, q1, q2)
        state4 = (q4, q1, q2, q3)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)
        action3 = agent3.choose_action(state3)
        action4 = agent4.choose_action(state4)

        q1_new = q1 + action1
        q2_new = q2 + action2
        q3_new = q3 + action3
        q4_new = q4 + action4

        reward1, reward2, reward3, reward4 = get_profits_three_firms(q1_new, q2_new, q3_new, q4_new, a, b, c1, c2, c3, c4)

        next_state1 = (q1_new, q2_new, q3_new, q4_new)
        next_state2 = (q2_new, q3_new, q4_new, q1_new)
        next_state3 = (q3_new, q4_new, q1_new, q2_new)
        next_state4 = (q4_new, q1_new, q2_new, q3_new)

        agent1.update(state1, action1, reward1, next_state1)
        agent2.update(state2, action2, reward2, next_state2)
        agent3.update(state3, action3, reward3, next_state3)
        agent4.update(state4, action4, reward4, next_state4)

        agent1.decay_epsilon()
        agent2.decay_epsilon()
        agent3.decay_epsilon()
        agent4.decay_epsilon()

        q1, q2, q3, q4 = q1_new, q2_new, q3_new, q4_new

    print()  # Add a newline after the progress output
    return agent1.history, agent2.history, agent3.history, agent4.history


# Example usage
if __name__ == "__main__":
    print("Running simulation. Current percentage complete:")

    # Start timing the simulation
    start_time = time.time()

    # Example usage for 4 firms
    q1_hist, q2_hist, q3_hist, q4_hist = run_cournot_simulation_three_firms(
        num_episodes=600000,
        max_q=25,  # Maximum quantity a firm can produce
        delta_n=2,  # Change in production allowed
        a=-1,  # Coefficient for price function (negative slope)
        b=40,  # Intercept for price function (base price when total quantity is 0)
        c1=5,  # Cost per unit for Firm 1
        c2=5,  # Cost per unit for Firm 2
        c3=5,  # Cost per unit for Firm 3        
        c4=5,  # Cost per unit for Firm 4
        alpha=0.1,  # Learning rate
        epsilon=1.0,  # Initial exploration rate
        epsilon_decay=0.995,  # Decay rate for exploration
        min_epsilon=0.05,  # Minimum exploration rate
        initial_q1=0,  # Initial quantity for Firm 1
        initial_q2=0,  # Initial quantity for Firm 2
        initial_q3=0,  # Initial quantity for Firm 3
        initial_q4=0
    )

    # Plot results for 4 firms with smoothing
    print("\nSmoothing data. Current percentage complete:")

    # Define the smoothing window size
    window_size = 5000

    # Initialize smoothing progress
    smoothing_steps = 4  # Number of firms being smoothed
    smoothing_count = 0

    # Apply moving average for smoothing
    q1_smooth = np.convolve(q1_hist, np.ones(window_size)/window_size, mode='valid')
    smoothing_count += 1
    print((smoothing_count * 100) // smoothing_steps, end=" ", flush=True)

    q2_smooth = np.convolve(q2_hist, np.ones(window_size)/window_size, mode='valid')
    smoothing_count += 1
    print((smoothing_count * 100) // smoothing_steps, end=" ", flush=True)

    q3_smooth = np.convolve(q3_hist, np.ones(window_size)/window_size, mode='valid')
    smoothing_count += 1
    print((smoothing_count * 100) // smoothing_steps, end=" ", flush=True)

    q4_smooth = np.convolve(q4_hist, np.ones(window_size)/window_size, mode='valid')
    smoothing_count += 1
    print((smoothing_count * 100) // smoothing_steps, end=" ", flush=True)

    print()  # Add a newline after smoothing progress

    # Plot the smoothed quantities
    plt.figure(figsize=(12, 6))
    plt.plot(q1_smooth, label="Firm 1 (smoothed) " + str(window_size), color="blue")
    plt.plot(q2_smooth, label="Firm 2 (smoothed) " + str(window_size), color="orange")
    plt.plot(q3_smooth, label="Firm 3 (smoothed) " + str(window_size), color="green")
    plt.plot(q4_smooth, label="Firm 4 (smoothed) " + str(window_size), color="black")

    # Add labels, title, and legend
    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title("Smoothed Cournot Learning Dynamics for 4 Firms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Calculate and display total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")

