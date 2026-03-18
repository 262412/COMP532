# GPU 检查和训练脚本
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import imageio
import os
from IPython.display import Image
import sys

print("=" * 60)
print("GPU 环境检查")
print("=" * 60)
print(f"CUDA 可用：{torch.cuda.is_available()}")
print(f"PyTorch 版本：{torch.__version__}")
print(f"Python 版本：{sys.version}")

if torch.cuda.is_available():
    print(f"✓ 可以使用 GPU 训练！")
    print(f"GPU 型号：{torch.cuda.get_device_name(0)}")
    print(f"GPU 数量：{torch.cuda.device_count()}")
    print(f"当前 GPU 索引：{torch.cuda.current_device()}")
else:
    print("✗ 未检测到 GPU，将使用 CPU 训练")
    
print("=" * 60)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n实际使用设备：{device}")
if device.type == "cuda":
    print(f"✓ 训练将在 GPU 上进行")
else:
    print(f"⚠ 训练将在 CPU 上进行")
print("=" * 60)

# 测试张量是否在正确的设备上
test_tensor = torch.randn(5).to(device)
print(f"测试张量设备：{test_tensor.device}")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

# Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99,
                 tau=0.005, buffer_size=100000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.losses = []

    def act(self, state, eval_mode=False):
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q = self.q_network(states).gather(1, actions)
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# Training
def train(env_name="LunarLander-v3", episodes=100):
    env = gym.make(env_name, render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    episode_rewards = []
    episode_lengths = []
    all_losses = []

    print(f"\n开始训练：{episodes} 个 episodes")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        all_losses.extend(agent.losses)
        agent.losses = []

        if (ep+1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {ep+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards, episode_lengths, all_losses

# Run training
print("\n开始训练...")
agent, rewards, lengths, losses = train(episodes=100)

print("\n" + "=" * 60)
print("训练完成！总结:")
print("=" * 60)
print(f"总 Episodes: {len(rewards)}")
print(f"平均奖励：{np.mean(rewards):.2f}")
print(f"最高奖励：{max(rewards):.2f}")
print(f"最后 100 集平均奖励：{np.mean(rewards[-100:]):.2f}")
print(f"平均步数：{np.mean(lengths):.1f}")
print(f"总训练步数：{sum(lengths)}")
print(f"设备：{device}")
if device.type == "cuda":
    print(f"✓ 使用了 GPU 加速")
    print(f"GPU 型号：{torch.cuda.get_device_name(0)}")
else:
    print(f"⚠ 使用 CPU 训练")
print("=" * 60)

# Plot
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(rewards, alpha=0.3, label='Episode Reward')
window = 10
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(rewards)), smoothed, 'r', label='Smoothed (10)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards')
plt.legend()

plt.subplot(1,2,2)
plt.plot(losses, alpha=0.5, label='Loss per update')
if len(losses) > 100:
    smoothed_loss = np.convolve(losses, np.ones(100)/100, mode='valid')
    plt.plot(range(99, len(losses)), smoothed_loss, 'r', label='Smoothed (100)')
plt.xlabel('Training Step')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_curves.png')
print("\n训练曲线已保存为 training_curves.png")
plt.show()


# Record a final episode
def record_episode(agent, env_name="LunarLander-v3", gif_path="lander_final.gif"):
    """
    记录智能体在训练后的表现
    """
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    state, _ = env.reset()
    done = False
    
    while not done:
        frames.append(env.render())
        action = agent.act(state, eval_mode=True)  # 使用eval模式，不加探索噪声
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    env.close()
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF 已保存到 {gif_path}")

print("\n录制最终表现...")
record_episode(agent)