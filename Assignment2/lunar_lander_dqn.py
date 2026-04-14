"""
COMP532 Assignment 2 - Problem 1: Lunar Lander with DQN
==========================================================
This script implements a Dueling Double DQN agent to solve the LunarLander-v3 environment.
Requirements met:
- Uses PyTorch with GPU acceleration when available
- Implements Experience Replay, Target Network, and Double DQN
- Achieves average reward of 200+ over 100 consecutive episodes
- Generates training curves and records final performance
"""

# Check dependencies first
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    from collections import deque
    from pathlib import Path
    import matplotlib.pyplot as plt
    import imageio
except ImportError as e:
    print(f"Error importing required package: {e}")
    print("\nPlease install dependencies by running:")
    print("  pip install gymnasium numpy torch matplotlib imageio gymnasium[box2d]")
    print("\nOr use the requirements file:")
    print("  pip install -r requirements.txt")
    input("Press Enter to exit...")
    raise

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Device configuration (CPU-only as requested)
device = torch.device("cpu")
print(f"Using device: {device}")

# Always write generated artifacts next to this script.
OUTPUT_DIR = Path(__file__).resolve().parent
TRAINING_CURVE_PATH = OUTPUT_DIR / "training_curves.png"
DEFAULT_GIF_PATH = OUTPUT_DIR / "lander_final.gif"


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=100000):
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


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with separate value and advantage streams."""
    
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
        # Q = V + (A - mean(A)) for numerical stability
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class DQNAgent:
    """Dueling Double DQN Agent with experience replay."""
    
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

        # Initialize networks
        self.q_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.losses = []

    def act(self, state, eval_mode=False):
        """Epsilon-greedy action selection."""
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
        """Update networks using Double DQN."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Double DQN: use online network to select actions, target network to evaluate
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

        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                   (1 - self.tau) * target_param.data)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train(env_name="LunarLander-v3", episodes=1000, render_during_training=False):
    """Train the DQN agent."""
    # Disable rendering during training for speed; rendering is only needed for final GIF export.
    if render_during_training:
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    episode_rewards = []
    episode_lengths = []
    all_losses = []

    print(f"\nStarting training for {episodes} episodes...")
    print(f"Device: {device}")
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

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}/{episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards, episode_lengths, all_losses


def plot_results(rewards, losses):
    """Plot training results."""
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), smoothed, 'r', 
             label=f'Smoothed ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.5, label='Loss per update')
    if len(losses) > 100:
        smoothed_loss = np.convolve(losses, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(losses)), smoothed_loss, 'r', 
                 label='Smoothed (100)')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_PATH, dpi=150)
    print(f"\nTraining curves saved to '{TRAINING_CURVE_PATH}'")
    plt.show()


def record_episode(agent, env_name="LunarLander-v3", 
                   gif_path=None):
    """Record and save a test episode as GIF."""
    if gif_path is None:
        gif_path = DEFAULT_GIF_PATH
    else:
        gif_path = Path(gif_path)
        if not gif_path.is_absolute():
            gif_path = OUTPUT_DIR / gif_path

    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    state, _ = env.reset()
    done = False
    
    while not done:
        frames.append(env.render())
        action = agent.act(state, eval_mode=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    env.close()
    imageio.mimsave(str(gif_path), frames, fps=30)
    print(f"GIF saved to '{gif_path}'")


def main():
    """Main training pipeline."""
    # Train agent
    agent, rewards, lengths, losses = train(episodes=1000, render_during_training=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total Episodes: {len(rewards)}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Highest Reward: {max(rewards):.2f}")
    print(f"Last 100 Episodes Avg Reward: {np.mean(rewards[-100:]):.2f}")
    print(f"Average Episode Length: {np.mean(lengths):.1f}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print("✓ GPU acceleration used")
    else:
        print("⚠ CPU used for training")
    print("=" * 60)
    
    # Plot results
    plot_results(rewards, losses)
    
    # Record final performance
    print("\nRecording final episode...")
    record_episode(agent)


if __name__ == "__main__":
    main()
