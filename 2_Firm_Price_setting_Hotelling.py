import numpy as np
import random
import matplotlib.pyplot as plt
import math

class BertrandAgent:
    def __init__(self, max_delta_p, alpha, epsilon, epsilon_decay, min_epsilon):
        self.max_delta_p = max_delta_p
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = {}
        self.history = []

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def get_legal_actions(self, current_price):
        deltas = np.arange(-self.max_delta_p, self.max_delta_p + 0.001, 0.05)
        return [round(current_price + d, 2) for d in deltas if current_price + d > 0]

    def choose_action(self, state):
        legal_actions = self.get_legal_actions(state)
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            q_values = [self.get_q(state, a) for a in legal_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q]
            action = random.choice(best_actions)
        self.history.append(action)
        return action

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        next_qs = [self.get_q(next_state, a) for a in self.get_legal_actions(next_state)]
        max_next_q = max(next_qs) if next_qs else 0.0
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def revenue(p1, p2, d1, d2, dist_type, mu_min=0.0, mu_max=1.0, mu_mean=1.0, mu_std=0.2,
            cap1=1.0, cap2=1.0, max_buyer_utility=5.0):
    mu_max_eff = min(
        mu_max,
        (max_buyer_utility - p1) / d1 if d1 > 0 else mu_max,
        (max_buyer_utility - p2) / d2 if d2 > 0 else mu_max
    )

    if mu_max_eff <= mu_min:
        return 0.0, 0.0, None

    if d1 == d2:
        mu_star = (mu_max + mu_min) / 2
    else:
        mu_star = (p2 - p1) / (d1 - d2)

    if dist_type == 'uniform':
        q1 = max(0, min(mu_star, mu_max_eff) - mu_min) / (mu_max - mu_min)
        q2 = max(0, mu_max_eff - max(mu_star, mu_min)) / (mu_max - mu_min)
    elif dist_type == 'normal':
        def norm_cdf(x):
            return 0.5 * (1 + math.erf((x - mu_mean) / (mu_std * math.sqrt(2))))
        q1 = max(0, norm_cdf(min(mu_star, mu_max_eff)) - norm_cdf(mu_min))
        q2 = max(0, norm_cdf(mu_max_eff) - norm_cdf(max(mu_star, mu_min)))
    else:
        raise ValueError("Invalid distribution type. Use 'uniform' or 'normal'.")

    q1 = min(q1, cap1)
    q2 = min(q2, cap2)

    r1 = p1 * q1
    r2 = p2 * q2
    return r1, r2, mu_star


def run_bertrand_qlearning(num_episodes, initial_price, max_delta_p, d1, d2,
                            dist_type='uniform', mu_min=0.0, mu_max=1.0,
                            mu_mean=1.0, mu_std=0.2,
                            cap1=1.0, cap2=1.0, alpha=0.1,
                            epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                            max_buyer_utility=5.0):

    agent1 = BertrandAgent(max_delta_p, alpha, epsilon, epsilon_decay, min_epsilon)
    agent2 = BertrandAgent(max_delta_p, alpha, epsilon, epsilon_decay, min_epsilon)

    price1 = initial_price
    price2 = initial_price

    mu_star_history = []
    r1_history = []
    r2_history = []

    for episode in range(num_episodes):
        state1 = price2
        state2 = price1

        price1 = agent1.choose_action(state1)
        price2 = agent2.choose_action(state2)

        r1, r2, mu_star = revenue(price1, price2, d1, d2, dist_type, mu_min, mu_max,
                                  mu_mean, mu_std, cap1, cap2, max_buyer_utility)

        agent1.update(state1, price1, r1, price2)
        agent2.update(state2, price2, r2, price1)

        agent1.decay_epsilon()
        agent2.decay_epsilon()

        mu_star_history.append(mu_star)

        r1_history.append(r1)
        r2_history.append(r2)

    return agent1.history, agent2.history, mu_star_history, r1_history, r2_history


if __name__ == "__main__":
    # Compute and print mean mu_star for the last 100 episodes
    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else float('nan')
    q1_hist, q2_hist, mu_star_hist, r1_hist, r2_hist = run_bertrand_qlearning(
        num_episodes=10000,
        initial_price=5,
        max_delta_p=0.2,
        d1=0.4,
        d2=0.6,
        dist_type='uniform',
        mu_min=0.6,
        mu_max=1.4,

        mu_mean=1.0,
        mu_std=0.3,
        cap1=1.0,
        cap2=1.0,
        alpha=0.1,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        max_buyer_utility=5.0
    )

    plt.plot(q1_hist, label="Firm 1 Price")
    plt.plot(q2_hist, label="Firm 2 Price")
    plt.xlabel("Episode")
    plt.ylabel("Price")
    plt.title("Results of Q-learning in Bertrand Competition")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Mean μ* (last 100 episodes):", round(safe_mean(mu_star_hist[-100:]),4))
    print("Mean p1 (last 100 episodes):", round(safe_mean(q1_hist[-100:]), 4))
    print("Mean p2 (last 100 episodes):", round(safe_mean(q2_hist[-100:]), 4))
    print("Mean r1 (last 100 episodes):", round(safe_mean(r1_hist[-100:]), 4))
    print("Mean r2 (last 100 episodes):", round(safe_mean(r2_hist[-100:]), 4))
    print("Mean p1 (last 100 episodes):", round(safe_mean(q1_hist[-100:]), 4))
    print("Mean p2 (last 100 episodes):", round(safe_mean(q2_hist[-100:]), 4))

    plt.plot(mu_star_hist, label="μ* (Indifferent Buyer)")
    plt.axhline(np.mean([m for m in mu_star_hist[-100:] if m is not None]), color='r', linestyle='--', label='Mean μ* (last 100)')
    plt.xlabel("Episode")
    plt.ylabel("μ*")
    plt.title("Evolution of Indifference Point Over Time")
    plt.ylim(0.6, 1.4)  # <-- Set y-axis limits here
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
