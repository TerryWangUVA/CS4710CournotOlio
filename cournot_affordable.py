import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # Import seaborn

class CournotAgent:
    def __init__(self, max_quantity, delta_n, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit):
        self.max_q = max_quantity
        self.delta_n = delta_n
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.affordable_housing_percentage = affordable_housing_percentage
        self.penalty_per_unit = penalty_per_unit

        self.action_space = list(range(-delta_n, delta_n + 1))  # change in production
        self.q_table = {}  # use dictionary due to large state space
        self.history = []
        self.affordable_history = [] # track affordable housing production

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

        # track the resulting quantity after the action
        self.history.append(state[0] + action)
        return action

    def update(self, state, action, reward, next_state, affordable_housing_provided):
        old_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in self.get_legal_actions(next_state)])
        new_q = old_q + self.alpha * (reward + max_next_q - old_q)  # include discounting
        self.q_table[(state, action)] = new_q
        self.affordable_history.append(affordable_housing_provided)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_legal_actions(self, state):
        current_q = state[0]
        return [a for a in self.action_space if 0 <= current_q + a <= self.max_q]

# economic parameters
def price_function(a, b, q_total):
    return a * q_total + b

def get_profits(q1, q2, a, b, c1, c2, affordable_housing_percentage, penalty_per_unit):
    P = price_function(a, b, q1 + q2)

    # calculate affordable housing requirements and penalties
    affordable_required_1 = q1 * affordable_housing_percentage
    affordable_required_2 = q2 * affordable_housing_percentage

    # assume all production is standard housing initially, no specific affordable housing production
    # we can be modified if you want to model specific affordable housing production decisions
    penalty1 = 0
    penalty2 = 0

    profit1 = P * q1 - c1 * q1 - penalty1
    profit2 = P * q2 - c2 * q2 - penalty2

    return profit1, profit2

def get_profits_three_firms(q1, q2, q3, a, b, c1, c2, c3, affordable_housing_percentage, penalty_per_unit):
    """Calculate profits for three firms, considering affordable housing requirements and penalties."""
    P = price_function(a, b, q1 + q2 + q3)

    # calculate affordable housing requirements
    affordable_required_1 = q1 * affordable_housing_percentage
    affordable_required_2 = q2 * affordable_housing_percentage
    affordable_required_3 = q3 * affordable_housing_percentage

    # calculate penalties if affordable housing requirements are not met.  assume no affordable housing is produced.
    penalty1 = max(0, affordable_required_1 * penalty_per_unit)
    penalty2 = max(0, affordable_required_2 * penalty_per_unit)
    penalty3 = max(0, affordable_required_3 * penalty_per_unit)

    profit1 = P * q1 - c1 * q1 - penalty1
    profit2 = P * q2 - c2 * q2 - penalty2
    profit3 = P * q3 - c3 * q3 - penalty3

    return profit1, profit2, profit3, affordable_required_1, affordable_required_2, affordable_required_3


def plot_quantities(q1_hist, q2_hist, q3_hist, title="Cournot Learning Dynamics (3 Firms)", smooth=True, window=500, save_path=None):
    """Plots the quantities produced by each firm over time, with optional smoothing."""
    plt.figure(figsize=(12, 6))

    # Smoothing using a moving average
    if smooth:
        q1_smooth = np.convolve(q1_hist, np.ones(window) / window, mode='same')  # Use 'same' for consistent length
        q2_smooth = np.convolve(q2_hist, np.ones(window) / window, mode='same')
        q3_smooth = np.convolve(q3_hist, np.ones(window) / window, mode='same')
        plt.plot(q1_smooth, label='Firm 1 (smoothed)')
        plt.plot(q2_smooth, label='Firm 2 (smoothed)')
        plt.plot(q3_smooth, label='Firm 3 (smoothed)')
    else:
        plt.plot(q1_hist, label='Firm 1')
        plt.plot(q2_hist, label='Firm 2')
        plt.plot(q3_hist, label='Firm 3')

    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_affordable_housing(aff1, aff2, aff3, title="Affordable Housing Requirements Over Time", save_path=None):
    """Plots the affordable housing requirements for each firm over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(aff1, label='Firm 1 - Affordable')
    plt.plot(aff2, label='Firm 2 - Affordable')
    plt.plot(aff3, label='Firm 3 - Affordable')
    plt.xlabel("Episode")
    plt.ylabel("Affordable Housing Required")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def export_q_table(agent, firm_name, top_n=10):
    """Exports the top N Q-values for a given agent to a CSV file."""
    # convert to DataFrame
    q_df = pd.DataFrame([
        {"State": k[0], "Action": k[1], "Q-Value": v}
        for k, v in agent.q_table.items()
    ])
    q_df = q_df.sort_values(by="Q-Value", ascending=False).head(top_n)
    q_df.to_csv(f"{firm_name}_top_q_values.csv", index=False)
    return q_df

def plot_convergence(q1_hist, q2_hist, q3_hist, window=1000, save_path=None):
    """Plots the moving average of quantities to visualize convergence."""
    q1_smooth = np.convolve(q1_hist, np.ones(window) / window, mode='valid')
    q2_smooth = np.convolve(q2_hist, np.ones(window) / window, mode='valid')
    q3_smooth = np.convolve(q3_hist, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(q1_smooth, label='Firm 1')
    plt.plot(q2_smooth, label='Firm 2')
    plt.plot(q3_smooth, label='Firm 3')
    plt.xlabel("Episode (Moving Average)")
    plt.ylabel("Quantity (Moving Average)")
    plt.title("Convergence of Quantities (Moving Average)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def run_cournot_simulation_three_firms(num_episodes, max_q, delta_n, a, b, c1, c2, c3,
                                       alpha, epsilon, epsilon_decay, min_epsilon,
                                       affordable_housing_percentage, penalty_per_unit,
                                       initial_q1=None, initial_q2=None, initial_q3=None):
    """Run Cournot simulation for three firms with affordable housing requirements."""
    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit)
    agent3 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit)

    # allow user to specify initial quantities
    q1 = initial_q1 if initial_q1 is not None else max_q // 3
    q2 = initial_q2 if initial_q2 is not None else max_q // 3
    q3 = initial_q3 if initial_q3 is not None else max_q // 3

    q1_hist = []
    q2_hist = []
    q3_hist = []
    affordable_hist_1 = []
    affordable_hist_2 = []
    affordable_hist_3 = []

    count = 0
    for episode in range(num_episodes):
        # calculate the percentage of progress
        percentage = ((episode + 1) * 100)  //  num_episodes

        # print percentage only when it changes (integer)
        if percentage > count:
            count = percentage
            print(count, end=" ", flush=True)

        state1 = (q1, q2, q3)
        state2 = (q2, q3, q1)
        state3 = (q3, q1, q2)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)
        action3 = agent3.choose_action(state3)

        q1_new = q1 + action1
        q2_new = q2 + action2
        q3_new = q3 + action3

        # get profits and affordable housing requirements
        reward1, reward2, reward3, affordable_required_1, affordable_required_2, affordable_required_3 = get_profits_three_firms(
            q1_new, q2_new, q3_new, a, b, c1, c2, c3, affordable_housing_percentage, penalty_per_unit
        )

        next_state1 = (q1_new, q2_new, q3_new)
        next_state2 = (q2_new, q3_new, q1_new)
        next_state3 = (q3_new, q1_new, q2_new)

        agent1.update(state1, action1, reward1, next_state1, affordable_required_1)
        agent2.update(state2, action2, reward2, next_state2, affordable_required_2)
        agent3.update(state3, action3, reward3, next_state3, affordable_required_3)

        agent1.decay_epsilon()
        agent2.decay_epsilon()
        agent3.decay_epsilon()

        q1, q2, q3 = q1_new, q2_new, q3_new

        q1_hist.append(q1)
        q2_hist.append(q2)
        q3_hist.append(q3)
        affordable_hist_1.append(affordable_required_1)
        affordable_hist_2.append(affordable_required_2)
        affordable_hist_3.append(affordable_required_3)

    print()  # add a newline after the progress output
    return q1_hist, q2_hist, q3_hist, agent1, agent2, agent3, affordable_hist_1, affordable_hist_2, affordable_hist_3


# Example usage
if __name__ == "__main__":

    print("Running simulation. Current percentage complete:")
    # example usage for 3 firms
    q1_hist, q2_hist, q3_hist, agent1, agent2, agent3, affordable_hist_1, affordable_hist_2, affordable_hist_3 = run_cournot_simulation_three_firms(
        num_episodes=1500000,
        max_q=20,  # maximum quantity a firm can produce
        delta_n=2,  # change in production allowed
        a=-1,  # coefficient for price function (negative slope)
        b=40,  # intercept for price function (base price when total quantity is 0)
        c1=6,  # cost per unit for Firm 1
        c2=6,  # cost per unit for Firm 2
        c3=6,  # cost per unit for Firm 3
        alpha=0.1,  # learning rate
        epsilon=1.0,  # initial exploration rate
        epsilon_decay=0.999995,  # decay rate for exploration
        min_epsilon=0.05,  # minimum exploration rate
        affordable_housing_percentage=0.2,  # 20% of production must be affordable
        penalty_per_unit=5,  # penalty for each unit of affordable housing not provided
        initial_q1=0,  # initial quantity for Firm 1
        initial_q2=0,  # initial quantity for Firm 2
        initial_q3=0   # initial quantity for Firm 3
    )

    # save plots to files
    plot_quantities(q1_hist, q2_hist, q3_hist, smooth=True, save_path="quantities_plot.png")
    plot_affordable_housing(affordable_hist_1, affordable_hist_2, affordable_hist_3, save_path="affordable_housing_plot.png") # think this var is wrong
    plot_convergence(q1_hist, q2_hist, q3_hist, save_path="convergence_plot.png")

    # export Q-tables to CSV
    export_q_table(agent1, "Firm1")
    export_q_table(agent2, "Firm2")
    export_q_table(agent3, "Firm3")


