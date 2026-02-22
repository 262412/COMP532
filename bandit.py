import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.normal(0, 1, k)   # true action values
        self.Q_est = np.zeros(k)                  # estimated action values
        self.N = np.zeros(k)                      # action counts

    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)

    def update_estimate(self, action, reward):
        self.N[action] += 1
        self.Q_est[action] += (reward - self.Q_est[action]) / self.N[action]

def run_experiment(epsilon, n_steps=2000, n_tasks=2000, k=10):
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
epsilons = [0, 0.01, 0.1]
results = {}
for eps in epsilons:
    print(f"Running epsilon = {eps}...")
    rewards, optimal = run_experiment(eps)
    results[eps] = (rewards, optimal)

# Plot results
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for eps in epsilons:
    plt.plot(results[eps][0], label=f'ε={eps}')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()

plt.subplot(1,2,2)
for eps in epsilons:
    plt.plot(results[eps][1], label=f'ε={eps}')
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.legend()

plt.tight_layout()
plt.savefig('figure1.png')
plt.show()