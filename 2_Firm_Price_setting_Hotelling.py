import numpy as np
import random
import matplotlib.pyplot as plt

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
        deltas = np.arange(-self.max_delta_p, self.max_delta_p + 0.001, 0.05)  # fewer, coarser steps
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
    if d1 == d2:
        q1 = 0.5
        q2 = 0.5
    else:
        mu_star = (p2 - p1) / (d1 - d2)

        if dist_type == 'uniform':
            F_mu_star = (mu_star - mu_min) / (mu_max - mu_min)
        elif dist_type == 'normal':
            F_mu_star = 0.5 * (1 + np.math.erf((mu_star - mu_mean) / (mu_std * np.sqrt(2))))
        else:
            raise ValueError("Invalid distribution type. Use 'uniform' or 'normal'.")

        F_mu_star = np.clip(F_mu_star, 0, 1)
        q1 = min(F_mu_star, cap1)
        q2 = min(1 - F_mu_star, cap2)

    # Participation constraint
    q1 = q1 if p1 + mu_mean * d1 <= max_buyer_utility else 0
    q2 = q2 if p2 + mu_mean * d2 <= max_buyer_utility else 0

    r1 = p1 * q1
    r2 = p2 * q2
    return r1, r2


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

    for episode in range(num_episodes):
        state1 = price2
        state2 = price1

        price1 = agent1.choose_action(state1)
        price2 = agent2.choose_action(state2)

        r1, r2 = revenue(price1, price2, d1, d2, dist_type, mu_min, mu_max,
                         mu_mean, mu_std, cap1, cap2, max_buyer_utility)

        agent1.update(state1, price1, r1, price2)
        agent2.update(state2, price2, r2, price1)

        agent1.decay_epsilon()
        agent2.decay_epsilon()

    return agent1.history, agent2.history


if __name__ == "__main__":
    q1_hist, q2_hist = run_bertrand_qlearning(
        num_episodes=20000,
        initial_price=1.0,
        max_delta_p=0.2,
        d1=0.2,
        d2=0.8,
        dist_type='uniform',
        mu_min=0.5,
        mu_max=1.5,
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
    plt.title("Price Learning with Faster Convergence Setup")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
