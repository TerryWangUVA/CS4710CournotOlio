import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""The Bertrand model is a model of competition in economics where firms produce the same good and compete by setting prices,
rather than quantities, leading to a Bertrand equilibrium where prices fall to marginal cost, mirroring perfect competition"""

class BertrandAgent:
    def __init__(self, min_price, max_price, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, cost_per_unit):
        self.min_price = min_price
        self.max_price = max_price
        self.delta_p = delta_p
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.affordable_housing_percentage = affordable_housing_percentage
        self.penalty_per_unit = penalty_per_unit
        self.cost_per_unit = cost_per_unit

        self.action_space = list(range(-delta_p, delta_p + 1))  # change in price
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

        # track the resulting price after the action
        self.history.append(state[0] + action)
        return action

    def update(self, state, action, reward, next_state, affordable_housing_provided):
        old_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in self.get_legal_actions(next_state)])
        new_q = old_q + self.alpha * (reward - old_q)  # no discounting
        self.q_table[(state, action)] = new_q
        self.affordable_history.append(affordable_housing_provided)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_legal_actions(self, state):
        current_price = state[0]
        return [a for a in self.action_space if self.min_price <= current_price + a <= self.max_price]

# Economic parameters
def demand_function(price, a, b):
    """Linear demand function: Q = a - b*P"""
    return max(0, a - b * price)  # ensure demand is non-negative

def get_profits_three_firms(p1, p2, p3, a, b, c1, c2, c3, affordable_housing_percentage, penalty_per_unit):
    """Calculate profits for three firms in a Bertrand model, considering affordable housing requirements and penalties."""

    # determine the lowest price
    min_price = min(p1, p2, p3)

    # calculate demand for each firm based on the lowest price
    if p1 == min_price:
        demand1 = demand_function(p1, a, b)
    else:
        demand1 = 0

    if p2 == min_price:
        demand2 = demand_function(p2, a, b)
    else:
        demand2 = 0

    if p3 == min_price:
        demand3 = demand_function(p3, a, b)
    else:
        demand3 = 0

    # if multiple firms have the lowest price, split the demand equally
    num_firms_at_min_price = sum([1 for p in [p1, p2, p3] if p == min_price])
    if num_firms_at_min_price > 1:
        total_demand = demand_function(min_price, a, b)
        demand1 = total_demand / num_firms_at_min_price if p1 == min_price else 0
        demand2 = total_demand / num_firms_at_min_price if p2 == min_price else 0
        demand3 = total_demand / num_firms_at_min_price if p3 == min_price else 0

    # calculate affordable housing requirements
    affordable_required_1 = demand1 * affordable_housing_percentage
    affordable_required_2 = demand2 * affordable_housing_percentage
    affordable_required_3 = demand3 * affordable_housing_percentage

    # calculate penalties if affordable housing requirements are not met.  Assume no affordable housing is produced.
    penalty1 = demand1 * affordable_housing_percentage * penalty_per_unit
    penalty2 = demand2 * affordable_housing_percentage * penalty_per_unit
    penalty3 = demand3 * affordable_housing_percentage * penalty_per_unit

    profit1 = (p1 - c1) * demand1 - penalty1
    profit2 = (p2 - c2) * demand2 - penalty2
    profit3 = (p3 - c3) * demand3 - penalty3

    return profit1, profit2, profit3, affordable_required_1, affordable_required_2, affordable_required_3


def plot_prices(p1_hist, p2_hist, p3_hist, title="Bertrand Learning Dynamics", smooth=False):
    plt.figure(figsize=(12, 6))

    if smooth:
        # moving average for smoothing
        window = 500
        p1_smooth = np.convolve(p1_hist, np.ones(window)/window, mode='valid')
        p2_smooth = np.convolve(p2_hist, np.ones(window)/window, mode='valid')
        p3_smooth = np.convolve(p3_hist, np.ones(window)/window, mode='valid')
        plt.plot(p1_smooth, label='Firm 1 (smoothed)')
        plt.plot(p2_smooth, label='Firm 2 (smoothed)')
        plt.plot(p3_smooth, label='Firm 3 (smoothed)')
    else:
        plt.plot(p1_hist, label='Firm 1')
        plt.plot(p2_hist, label='Firm 2')
        plt.plot(p3_hist, label='Firm 3')

    plt.xlabel("Episode")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_bertrand_simulation_three_firms(num_episodes, min_price, max_price, delta_p, a, b, c1, c2, c3,
                                        alpha, epsilon, epsilon_decay, min_epsilon,
                                        affordable_housing_percentage, penalty_per_unit,
                                        initial_p1=None, initial_p2=None, initial_p3=None):
    """Run Bertrand simulation for three firms with affordable housing requirements."""
    agent1 = BertrandAgent(min_price, max_price, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, c1)
    agent2 = BertrandAgent(min_price, max_price, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, c2)
    agent3 = BertrandAgent(min_price, max_price, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, c3)

    # allow user to specify initial prices
    p1 = initial_p1 if initial_p1 is not None else (min_price + max_price) // 2
    p2 = initial_p2 if initial_p2 is not None else (min_price + max_price) // 2
    p3 = initial_p3 if initial_p3 is not None else (min_price + max_price) // 2

    p1_hist = []
    p2_hist = []
    p3_hist = []
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

        state1 = (p1, p2, p3)
        state2 = (p2, p3, p1)
        state3 = (p3, p1, p2)

        action1 = agent1.choose_action(state1)
        action2 = agent2.choose_action(state2)
        action3 = agent3.choose_action(state3)

        p1_new = p1 + action1
        p2_new = p2 + action2
        p3_new = p3 + action3

        # get profits and affordable housing requirements
        reward1, reward2, reward3, affordable_required_1, affordable_required_2, affordable_required_3 = get_profits_three_firms(
            p1_new, p2_new, p3_new, a, b, c1, c2, c3, affordable_housing_percentage, penalty_per_unit
        )

        next_state1 = (p1_new, p2_new, p3_new)
        next_state2 = (p2_new, p3_new, p1_new)
        next_state3 = (p3_new, p1_new, p2_new)

        agent1.update(state1, action1, reward1, next_state1, affordable_required_1)
        agent2.update(state2, action2, reward2, next_state2, affordable_required_2)
        agent3.update(state3, action3, reward3, next_state3, affordable_required_3)

        agent1.decay_epsilon()
        agent2.decay_epsilon()
        agent3.decay_epsilon()

        p1, p2, p3 = p1_new, p2_new, p3_new

        p1_hist.append(p1)
        p2_hist.append(p2)
        p3_hist.append(p3)
        affordable_hist_1.append(affordable_required_1)
        affordable_hist_2.append(affordable_required_2)
        affordable_hist_3.append(affordable_required_3)

    print()  # add a newline after the progress output
    return p1_hist, p2_hist, p3_hist, affordable_hist_1, affordable_hist_2, affordable_hist_3


if __name__ == "__main__":

    print("Running simulation. Current percentage complete:")
    # example usage for 3 firms
    p1_hist, p2_hist, p3_hist, affordable_hist_1, affordable_hist_2, affordable_hist_3 = run_bertrand_simulation_three_firms(
        num_episodes=1500000,
        min_price=1,  # minimum price a firm can set
        max_price=40,  # maximum price a firm can set
        delta_p=1,  # change in price allowed
        a=100,  # intercept for demand function
        b=2,    # slope for demand function
        c1=6,  # cost per unit for Firm 1
        c2=6,  # cost per unit for Firm 2
        c3=6,  # cost per unit for Firm 3
        alpha=0.1,  # learning rate
        epsilon=1.0,  # initial exploration rate
        epsilon_decay=0.995,  # decay rate for exploration
        min_epsilon=0.05,  # minimum exploration rate
        affordable_housing_percentage=0.2,  # 20% of production must be affordable
        penalty_per_unit=2,  # penalty for each unit of affordable housing not provided
        initial_p1=20,  # initial price for Firm 1
        initial_p2=20,  # initial price for Firm 2
        initial_p3=20   # initial price for Firm 3
    )

