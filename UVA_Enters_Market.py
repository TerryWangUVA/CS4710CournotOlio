import numpy as np
import random
import matplotlib.pyplot as plt

class CournotAgent:
    def __init__(self, max_quantity, delta_n, alpha, epsilon, epsilon_decay, min_epsilon):
        self.max_q = max_quantity
        self.delta_n = delta_n
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.action_space = list(range(-delta_n, delta_n + 1))
        self.q_table = {}
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
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_legal_actions(self, state):
        current_q = state[0]
        return [a for a in self.action_space if 0 <= current_q + a <= self.max_q]

def price_function(a, b, q_total):
    return a * q_total + b

def get_profits_three_firms(q1, q2, q3, a, b, c1, c2, c3):
    P = price_function(a, b, q1 + q2 + q3)
    profit1 = P * q1 - c1 * q1
    profit2 = P * q2 - c2 * q2
    profit3 = P * q3 - c3 * q3
    return profit1, profit2, profit3

# Main simulation loop for scenario 3 (UVA as fixed third firm)
def run_cournot_simulation_with_uva(num_episodes, max_q, delta_n, a, b, c1, c2, c3,
                                    alpha, epsilon, epsilon_decay, min_epsilon,
                                    Q_UVA, initial_q1=None, initial_q2=None):
    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)

    q1 = initial_q1 if initial_q1 is not None else max_q // 2
    q2 = initial_q2 if initial_q2 is not None else max_q // 2
    q3 = Q_UVA

    count = 0
    for episode in range(num_episodes):
        percentage = ((episode + 1) * 100)
        if percentage > count:
            count = percentage
            print(count, end=" ", flush=True)

        state1 = (q1, q2, q3)
        state2 = (q2, q1, q3)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)

        q1_new = q1 + action1
        q2_new = q2 + action2
        q3 = Q_UVA  # Fixed UVA supply

        reward1, reward2, _ = get_profits_three_firms(q1_new, q2_new, q3, a, b, c1, c2, c3)

        next_state1 = (q1_new, q2_new, q3)
        next_state2 = (q2_new, q1_new, q3)

        agent1.update(state1, action1, reward1, next_state1)
        agent2.update(state2, action2, reward2, next_state2)

        agent1.decay_epsilon()
        agent2.decay_epsilon()

        q1, q2 = q1_new, q2_new

    print()
    return agent1.history, agent2.history, [Q_UVA] * num_episodes

def plot_quantities_three_firms(q1_hist, q2_hist, q3_hist, title="Cournot Learning with UVA", window=500):
    q1_smooth = np.convolve(q1_hist, np.ones(window)/window, mode='valid')
    q2_smooth = np.convolve(q2_hist, np.ones(window)/window, mode='valid')
    q3_smooth = q3_hist[window-1:]

    plt.figure(figsize=(12, 6))
    plt.plot(q1_smooth, label="Firm 1 (smoothed)")
    plt.plot(q2_smooth, label="Firm 2 (smoothed)")
    plt.plot(q3_smooth, label="UVA (fixed)", linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Scenario 3: UVA enters market with fixed quantity")

    q1_hist, q2_hist, q3_hist = run_cournot_simulation_with_uva(
        num_episodes=200000,
        max_q=20,
        delta_n=2,
        a=-1,
        b=40,
        c1=6,
        c2=6,
        c3=6,         # Not used for UVA since supply is fixed, but required by profit function
        alpha=0.1,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        Q_UVA=10      # UVA's fixed market supply
    )

    plot_quantities_three_firms(q1_hist, q2_hist, q3_hist, title="Scenario 3: UVA Enters Market", window=500)

# UVA Supply Shifts Market Competition
# UVA adds 10 units of supply regardless of what the firms do.
# This increases total market supply, which lowers the price (because price depends on total quantity: 

# So the two firms have less incentive to produce large quantities, as profits are squeezed.
# Result: They learn to produce less than they would in a normal 2-firm Cournot setup.

# In the 2 Firm setup the 2 firms produce at 10-12, while here its much lower due to the UVA supply.
# This is consistent with Cournot equilibrium logic: a public supplier with fixed output and zero/low cost crowds out some private production.