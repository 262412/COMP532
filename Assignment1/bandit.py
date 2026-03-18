import numpy as np
import matplotlib.pyplot as plt

DEFAULT_K = 10
K_VALUES = [5, 10, 20]

class Bandit:
    def __init__(self, k=DEFAULT_K):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)   # true action values
        self.Q_est = np.zeros(k)                  # estimated action values
        self.N = np.zeros(k)                      # action counts

    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)

    def update_estimate(self, action, reward):
        self.N[action] += 1
        self.Q_est[action] += (reward - self.Q_est[action]) / self.N[action]

def run_experiment(epsilon, n_steps=2000, n_tasks=2000, k=DEFAULT_K):
    rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)
    for _ in range(n_tasks):
        bandit = Bandit(k)
        optimal = np.argmax(bandit.q_true)
        for step in range(n_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else:
                # greedy with random tie-breaking
                max_val = np.max(bandit.Q_est)
                candidates = np.where(bandit.Q_est == max_val)[0]
                action = np.random.choice(candidates)
            reward = bandit.get_reward(action)
            bandit.update_estimate(action, reward)
            rewards[step] += reward
            if action == optimal:
                optimal_actions[step] += 1
    rewards /= n_tasks
    optimal_actions /= n_tasks
    optimal_actions *= 100
    return rewards, optimal_actions

# Run experiments
k_values = K_VALUES
epsilons = [0, 0.01, 0.1]
results = {}
for k in k_values:
    results[k] = {}
    for eps in epsilons:
        print(f"Running k = {k}, epsilon = {eps}...")
        rewards, optimal = run_experiment(eps, k=k)
        results[k][eps] = (rewards, optimal)

# Plot results (one figure per k)
for k in k_values:
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    for eps in epsilons:
        plt.plot(results[k][eps][0], label=f'ε={eps}')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.title(f'Average reward (k={k})')
    plt.legend()

    plt.subplot(1,2,2)
    for eps in epsilons:
        plt.plot(results[k][eps][1], label=f'ε={eps}')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.title(f'Optimal action % (k={k})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'figure_k{k}.png')
    plt.show()
