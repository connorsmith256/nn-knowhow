import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gridworld import GridWorld

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=10_000)
        self.tau = 0.001
        self.gamma = 0.995
        self.eps = 1.0
        self.eps_min = 0.02
        self.eps_decay = 0.998
        self.lr = 5e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(agent.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        cur_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(dim=1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # loss = F.mse_loss(cur_q_values, target_q_values.detach())
        loss = F.smooth_l1_loss(cur_q_values, target_q_values.detach())
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def best_action(self, state, width, height):
        q_values = agent.model(torch.tensor(state_to_tensor(state, width, height), dtype=torch.float32).to(agent.device))
        best_action = q_values.argmax().item()
        return best_action

def state_to_tensor(state, width, height):
    x, y = state % width, state // width
    return [x / (width - 1), y / (height - 1)]

def train(env, agent, episodes, batch_size):
    total_rewards = np.zeros(episodes)
    total_done = np.zeros(episodes)
    total_steps = np.zeros(episodes)
    for episode in range(episodes):
        state, _ = env.reset()
        state = state_to_tensor(state, env.width, env.height)
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = state_to_tensor(next_state, env.width, env.height)
            agent.remember(state, action, reward, next_state, done or truncated)
            state = next_state
            total_rewards[episode] += reward
            total_done[episode] = done
            total_steps[episode] = env.steps if done else 0

            agent.replay(batch_size)
            agent.update_target_model()

        if agent.eps > agent.eps_min:
            agent.eps *= agent.eps_decay

        log_every = 10
        if episode >= log_every and episode % log_every == 0:
            avg_reward = np.mean(total_rewards[(episode - log_every):episode + 1])
            avg_done = np.mean(total_done[(episode - log_every):episode + 1]) * 100
            finished_episodes = (episode - log_every) + np.flatnonzero(total_steps[(episode - log_every):episode + 1])
            avg_steps = int(np.mean(total_steps[finished_episodes])) if len(finished_episodes) > 0 else 0
            print(f"Episode: {episode:4d}, Avg Reward: {avg_reward:.2f}, Avg Done: {avg_done:.2f}%, Avg Steps: {avg_steps if avg_steps > 0 else 'N/A'}, Eps: {agent.eps:.3f}, ")

        draw_every = 50
        if episode % draw_every == 0:
            env.print_policy(agent)

for seed in [1, 2, 4, 8, 16]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f'SEED: {seed}')

    env = GridWorld(20, 20, num_obstacles=20, max_steps=500)
    agent = DQNAgent(2, env.action_space.n)
    train(env, agent, int(2e3), 128)
    env.print_policy(agent)
