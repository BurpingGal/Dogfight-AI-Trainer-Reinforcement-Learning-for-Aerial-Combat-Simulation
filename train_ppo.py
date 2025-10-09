# Install dependencies first
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install("gymnasium")
install("torch")
install("numpy")
install("jsbsim")  # Add this line

# Now import
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
from azureml.core import Run

# Get Azure ML run context
run = Run.get_context()

# Import Azure ML at the top, but use after initialization
from azureml.core import Run

# Get Azure ML run context (must be after imports)
run = Run.get_context()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
LR = 0.0003
GAMMA = 0.99
EPISODES = 1000
MAX_TIMESTEPS = 500
UPDATE_EVERY = 2000
CLIP_EPS = 0.2

# Import environment after dependencies
from src.environment.dogfight_env import DogfightEnv

# Create environment to get dimensions
env = DogfightEnv()
state_dim = env.observation_space.shape[0]  # Should be 17
action_dim = env.action_space.shape[0]      # Should be 4
print(f"State dim: {state_dim}, Action dim: {action_dim}")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.critic(x)
        mean = torch.tanh(self.actor_mean(x))
        std = torch.exp(self.actor_log_std).expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

# Create agent and optimizer
agent = ActorCritic().to(device)
optimizer = optim.Adam(agent.parameters(), lr=LR)

# Load saved model if exists
if os.path.exists('dogfight_ppo_agent.pth'):
    agent.load_state_dict(torch.load('dogfight_ppo_agent.pth', map_location=device))
    print("Model loaded successfully.")
else:
    print("No saved model found. Starting from scratch.")

# Storage
states = []
actions = []
log_probs = []
rewards = []
dones = []
values = []

# Training loop
def compute_returns(next_value, dones, rewards, values, gamma=GAMMA):
    R = next_value
    returns = []
    for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
    return returns

print("Starting PPO training...")

for episode in range(EPISODES):
    state, _ = env.reset()
    ep_reward = 0  # ← This is your episode reward

    for t in range(MAX_TIMESTEPS):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, value = agent(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        next_state, reward, done, trunc, _ = env.step(action.cpu().numpy())

        # Tactical feedback
        radar = next_state[-5:-1]
        missile_alert = next_state[-1]
        tactic = "DEFENSIVE_SPIRAL" if missile_alert > 0.5 else \
                 "TURN_FIGHT" if (radar[2] < 1500 and abs(radar[0]) < 15) else \
                 "BOOM_AND_ZOOM" if (radar[3] > 100 and radar[2] > 3000) else "SEARCH"
        print(f"Tactic: {tactic}, Range: {radar[2]:.0f}m, Closure: {radar[3]:.1f}m/s, Missile: {missile_alert}")

        states.append(state)
        actions.append(action.cpu().numpy())
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done or trunc)
        values.append(value)

        state = next_state
        ep_reward += reward

        if (len(rewards) % UPDATE_EVERY == 0) or done or trunc:
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
            log_probs_old = torch.cat(log_probs, dim=0).detach().to(device)
            next_value = agent(torch.FloatTensor(state).unsqueeze(0).to(device))[1].detach()
            returns = compute_returns(next_value, dones, rewards, values, GAMMA)
            returns = torch.FloatTensor(returns).to(device)
            values_tensor = torch.cat(values, dim=0).detach().to(device)

            advantages = returns - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(3):
                dist_new, values_new = agent(states_tensor)
                log_probs_new = dist_new.log_prob(actions_tensor).sum(dim=-1, keepdim=True)
                ratio = (log_probs_new - log_probs_old).exp()

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values_new, returns)
                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ✅ Clear buffers
            states.clear()
            actions.clear()
            log_probs.clear()
            rewards.clear()
            dones.clear()
            values.clear()

        if done or trunc:
            break

    # ✅ LOG METRICS HERE — after ep_reward is calculated
    run.log("episode_reward", ep_reward)
    run.log("tactic_used", tactic)
    run.log("training_loss", loss.item() if 'loss' in locals() else 0.0)

    # Optional: print every 10 episodes
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {ep_reward:.2f}")

# Save model
torch.save(agent.state_dict(), 'dogfight_ppo_agent.pth')
print("Model saved successfully.")
