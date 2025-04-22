import numpy as np
import random
import matplotlib.pyplot as plt
import time


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


def price_function(a, b, q_total):
    return a * q_total + b


def get_profits(quantities, a, b, costs):
    """Calculate profits for all firms."""
    q_total = sum(quantities)
    P = price_function(a, b, q_total)
    profits = [P * q - c * q for q, c in zip(quantities, costs)]
    return profits


def calculate_equilibrium(num_firms, a, b, costs):
    """Calculate the theoretical equilibrium quantities and market price."""
    # Assume all firms have the same cost
    c = costs[0]  # Cost per unit for each firm (assumes symmetry)

    # Calculate equilibrium quantity for each firm
    q_i = (b - c) / (-1 * (a * (num_firms + 1)))

    # Calculate total quantity and market price
    Q_total = num_firms * q_i
    P = a * Q_total + b

    return q_i, Q_total, P


def run_cournot_simulation(num_firms, num_episodes, max_q, delta_n, a, b, costs, 
                           alpha, epsilon, epsilon_decay, min_epsilon, initial_quantities=None):
    """Run Cournot simulation for any number of firms."""
    agents = [CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon) for _ in range(num_firms)]

    # Initialize quantities
    quantities = initial_quantities if initial_quantities else [max_q // num_firms] * num_firms

    histories = [[] for _ in range(num_firms)]

    start_time = time.time()  # Start timing the simulation

    for episode in range(num_episodes):
        # Calculate the percentage of progress
        if episode % (num_episodes // 100) == 0:
            elapsed_time = (time.time() - start_time) / 60  # Elapsed time in minutes
            print(f"{(episode / num_episodes) * 100:.1f}% Complete | Elapsed Time: {elapsed_time:.2f} minutes", end="\r", flush=True)

        # Create states for each firm
        states = [tuple(quantities[i:] + quantities[:i]) for i in range(num_firms)]

        # Each agent chooses an action
        actions = [agent.choose_action(state) for agent, state in zip(agents, states)]

        # Update quantities based on actions
        new_quantities = [q + a for q, a in zip(quantities, actions)]

        # Calculate rewards (profits)
        rewards = get_profits(new_quantities, a, b, costs)

        # Update Q-tables for each agent
        new_states = [tuple(new_quantities[i:] + new_quantities[:i]) for i in range(num_firms)]
        for agent, state, action, reward, next_state in zip(agents, states, actions, rewards, new_states):
            agent.update(state, action, reward, next_state)

        # Decay epsilon for each agent
        for agent in agents:
            agent.decay_epsilon()

        # Update quantities and record histories
        quantities = new_quantities
        for i, q in enumerate(quantities):
            histories[i].append(q)

    print("\n100.0% Complete")
    return histories


if __name__ == "__main__":
    # Simulation parameters
    num_firms = 5  # Number of firms participating in the market
    num_episodes = 15000000  # Number of episodes (iterations) for the simulation
    max_q = 30  # Maximum quantity a firm can produce
    delta_n = 2  # Maximum change in production allowed per action
    a = -1  # Slope of the price function (negative value, price decreases as total quantity increases)
    b = 60  # Intercept of the price function (base price when total quantity is 0)
    costs = [6] * num_firms  # Cost per unit for each firm (assumes all firms have the same cost)
    alpha = 0.1  # Learning rate for Q-learning (how quickly agents update their Q-values)
    epsilon = 1.0  # Initial exploration rate (probability of choosing a random action)
    epsilon_decay = 0.995  # Decay rate for exploration (reduces epsilon after each episode)
    min_epsilon = 0.05  # Minimum exploration rate (ensures some exploration continues)
    initial_quantities = [0] * num_firms  # Initial production quantities for all firms


    # Calculate theoretical equilibrium
    q_i, Q_total, P = calculate_equilibrium(num_firms, a, b, costs)
    print(f"Theoretical Equilibrium per Firm: {q_i:.2f}")
    print(f"Theoretical Total Quantity: {Q_total:.2f}")
    print(f"Theoretical Market Price: {P:.2f}")

    print("Running simulation...")
    start_time = time.time()

    # Run the simulation
    histories = run_cournot_simulation(num_firms, num_episodes, max_q, delta_n, a, b, costs, 
                                       alpha, epsilon, epsilon_decay, min_epsilon, initial_quantities)

    # Plot results
    print("\nSmoothing data and plotting results...")
    window_size = 40000
    plt.figure(figsize=(13, 7))

    # Calculate the total number of points to be smoothed across all firms
    total_points = sum(len(history) - window_size + 1 for history in histories)  # Total points after smoothing
    points_per_percent = total_points // 100  # Points required for each 1% progress
    current_points = 0  # Track processed points
    progress = 0  # Track progress percentage

    # Smooth and plot each firm's data
    for i, history in enumerate(histories):
        smooth_history = np.convolve(history, np.ones(window_size) / window_size, mode='valid')
        for _ in range(len(smooth_history)):
            current_points += 1
            # Update progress percentage when the threshold is reached
            if current_points // points_per_percent > progress:
                progress = current_points // points_per_percent
                print(f"Smoothing Progress: {progress}%", end="\r", flush=True)
        plt.plot(smooth_history, label=f"Firm {i + 1} (smoothed)" + str(window_size), linewidth=.8)  # Set linewidth to 1

    print("\nSmoothing Complete")


    # Output the last 10 values for each firm
    print("\nLast 10 values for each firm:")
    for i, history in enumerate(histories):
        print(f"Firm {i + 1}: {history[-10:]}")

    # Calculate the average of the averages of the last 100 values for each firm
    average_of_averages = 0
    print("\nAverage of the last 100 values for each firm:")
    for i, history in enumerate(histories):
        last_100_avg = np.mean(history[-100:]) if len(history) >= 100 else np.mean(history)
        print(f"Firm {i + 1}: {last_100_avg:.2f}")
        average_of_averages += last_100_avg

    average_of_averages /= len(histories)
    print(f"\nAverage of the averages of the last 100 values: {average_of_averages:.2f}")

    # Calculate and display total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed_time / 60:.2f} minutes")
    
    
    # Add labels, title, and legend
    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title(f"Smoothed Cournot Learning Dynamics for {num_firms} Firms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()
