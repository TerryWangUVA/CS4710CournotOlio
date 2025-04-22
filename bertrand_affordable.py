import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BertrandAgent:
    def __init__(self, price_range, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, marginal_cost):
        self.min_price, self.max_price = price_range
        self.delta_p = delta_p
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.affordable_housing_percentage = affordable_housing_percentage
        self.penalty_per_unit = penalty_per_unit
        self.cost = marginal_cost

        self.action_space = list(range(self.min_price, self.max_price + 1))
        self.q_table = {}
        self.history = []
        self.affordable_history = []

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            q_values = [self.get_q(state, a) for a in self.action_space]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.action_space, q_values) if q == max_q]
            action = random.choice(best_actions)

        self.history.append(action)
        return action

    def update(self, state, action, reward, next_state, affordable_housing_provided):
        old_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in self.action_space])
        new_q = old_q + self.alpha * (reward + max_next_q - old_q)
        self.q_table[(state, action)] = new_q
        self.affordable_history.append(affordable_housing_provided)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# demand and profit logic
def get_demand(p1, p2, p3, total_demand=300):
    """Split demand based on price (lower price = more demand)."""
    prices = [p1, p2, p3]
    min_price = min(prices)
    counts = prices.count(min_price)

    if counts == 1:
        demand = [0, 0, 0]
        winner = prices.index(min_price)
        demand[winner] = total_demand
    else:
        split = total_demand // counts
        demand = [split if p == min_price else 0 for p in prices]

    return demand  # list of demands per firm

def get_profits(prices, demands, costs, affordable_housing_percentage, penalty_per_unit):
    profits = []
    aff_requirements = []

    for p, q, c in zip(prices, demands, costs):
        affordable_required = q * affordable_housing_percentage
        penalty = affordable_required * penalty_per_unit  # assume they fail to meet
        profit = p * q - c * q - penalty
        profits.append(profit)
        aff_requirements.append(affordable_required)

    return profits, aff_requirements


def plot_prices(p1_hist, p2_hist, p3_hist, title="Bertrand Learning Dynamics", smooth=True, window=500):
    plt.figure(figsize=(12, 6))
    if smooth:
        p1_smooth = np.convolve(p1_hist, np.ones(window) / window, mode='same')
        p2_smooth = np.convolve(p2_hist, np.ones(window) / window, mode='same')
        p3_smooth = np.convolve(p3_hist, np.ones(window) / window, mode='same')
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

# main simulation
def run_bertrand_simulation(num_episodes, price_range, delta_p, alpha, epsilon, epsilon_decay, min_epsilon,
                            affordable_housing_percentage, penalty_per_unit, costs):
    agent1 = BertrandAgent(price_range, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, costs[0])
    agent2 = BertrandAgent(price_range, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, costs[1])
    agent3 = BertrandAgent(price_range, delta_p, alpha, epsilon, epsilon_decay, min_epsilon, affordable_housing_percentage, penalty_per_unit, costs[2])

    p1_hist, p2_hist, p3_hist = [], [], []
    aff1_hist, aff2_hist, aff3_hist = [], [], []

    for episode in range(num_episodes):
        state1 = (0,)  # could be demand in prior round or competitor prices
        state2 = (0,)
        state3 = (0,)

        p1 = agent1.choose_action(state1)
        p2 = agent2.choose_action(state2)
        p3 = agent3.choose_action(state3)

        prices = [p1, p2, p3]
        demands = get_demand(p1, p2, p3)
        profits, aff_reqs = get_profits(prices, demands, costs, affordable_housing_percentage, penalty_per_unit)

        next_state = (0,)  # place holder

        agent1.update(state1, p1, profits[0], next_state, aff_reqs[0])
        agent2.update(state2, p2, profits[1], next_state, aff_reqs[1])
        agent3.update(state3, p3, profits[2], next_state, aff_reqs[2])

        agent1.decay_epsilon()
        agent2.decay_epsilon()
        agent3.decay_epsilon()

        p1_hist.append(p1)
        p2_hist.append(p2)
        p3_hist.append(p3)

        aff1_hist.append(aff_reqs[0])
        aff2_hist.append(aff_reqs[1])
        aff3_hist.append(aff_reqs[2])

    plot_prices(p1_hist, p2_hist, p3_hist)
    return agent1, agent2, agent3

# Example run
run_bertrand_simulation(
    num_episodes=10000,
    price_range=(50, 150),  # allowable prices
    delta_p=1,
    alpha=0.1,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    affordable_housing_percentage=0.2,
    penalty_per_unit=5,
    costs=[60, 65, 70]  # marginal costs per firm
)
