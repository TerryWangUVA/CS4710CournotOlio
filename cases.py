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


# SCENARIO-SPECIFIC PROFIT FUNCTIONS

def price_function(a, b, q_total):
    return a * q_total + b


def get_profits(q1, q2, a, b, c1, c2, scenario='baseline', r=0.1, P_affordable=10, Q_UVA=0):
    """Calculate profits based on scenario"""

    if scenario == "uva":
        total_q = q1 + q2 + Q_UVA
    else:
        total_q = q1 + q2

    P = price_function(a, b, total_q)

    if scenario == "baseline":
        profit1 = (P - c1) * q1
        profit2 = (P - c2) * q2

    elif scenario == "affordable":
        # Firms are required to allocate r% of their supply to lower-cost units
        profit1 = ((P - c1) * q1) - (r * q1 * P_affordable)
        profit2 = ((P - c2) * q2) - (r * q2 * P_affordable)

    elif scenario == "uva":
        profit1 = (P - c1) * q1
        profit2 = (P - c2) * q2

    else:
        raise ValueError("Invalid scenario specified.")

    return profit1, profit2


# MAIN SIMULATION LOOP

def run_cournot_simulation(num_episodes, max_q, delta_n, a, b, c1, c2,
                           alpha, epsilon, epsilon_decay, min_epsilon,
                           initial_q1=None, initial_q2=None,
                           scenario='baseline', r=0.1, P_affordable=10, Q_UVA=0):

    agent1 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = CournotAgent(max_q, delta_n, alpha, epsilon, epsilon_decay, min_epsilon)

    q1 = max_q // 2 if initial_q1 is None else initial_q1
    q2 = max_q // 2 if initial_q2 is None else initial_q2

    for episode in range(num_episodes):
        state1 = (q1, q2)
        state2 = (q2, q1)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)

        q1_new = q1 + action1
        q2_new = q2 + action2

        reward1, reward2 = get_profits(
            q1_new, q2_new, a, b, c1, c2,
            scenario=scenario, r=r, P_affordable=P_affordable, Q_UVA=Q_UVA
        )

        next_state1 = (q1_new, q2_new)
        next_state2 = (q2_new, q1_new)

        agent1.update(state1, action1, reward1, next_state1)
        agent2.update(state2, action2, reward2, next_state2)

        agent1.decay_epsilon()
        agent2.decay_epsilon()

        q1, q2 = q1_new, q2_new

    return agent1.history, agent2.history


# PLOTTING UTILITY

def plot_quantities(q1_hist, q2_hist, title="Cournot Learning Dynamics", smooth=False):
    plt.figure(figsize=(12, 6))
    if smooth:
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

#Example usage

# scenario 1: baseline
q1_hist, q2_hist = run_cournot_simulation(
    num_episodes=200000,
    max_q=20,
    delta_n=2,
    a=-1,
    b=40,
    c1=4,
    c2=4,
    alpha=0.1,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    scenario='baseline'
)
plot_quantities(q1_hist, q2_hist, title="scenario 1: baseline", smooth=True)


# scenario 2: affordable housing
q1_aff, q2_aff = run_cournot_simulation(
    num_episodes=200000,
    max_q=20,
    delta_n=2,
    a=-1,
    b=40,
    c1=4,
    c2=4,
    alpha=0.1,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    scenario='affordable',
    r=0.3,               # assuming 30% must be affordable
    P_affordable=10      # assuming government-subsidized price
)
plot_quantities(q1_aff, q2_aff, title="scenario 2: affordable housing", smooth=True)


# scenario 3: UVA enters market
q1_uva, q2_uva = run_cournot_simulation(
    num_episodes=200000,
    max_q=20,
    delta_n=2,
    a=-1,
    b=40,
    c1=4,
    c2=4,
    alpha=0.1,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    scenario='uva',
    Q_UVA=10  # supply added by UVA
)
plot_quantities(q1_uva, q2_uva, title="scenario 3: UVA enters market", smooth=True)


#scenario 4, affordable housing AND UVA enters market (only start after finishing first 3 scenarios)
