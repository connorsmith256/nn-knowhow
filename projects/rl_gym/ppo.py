import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gridworld import GridWorld

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.epochs = 16
        self.lr = 1e-3
        self.gamma = 0.99
        self.eps = 0.2
        self.entropy_coef = 0.1
        self.value_coef = 0.5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_probs, _ = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, old_log_probs, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, values = self.model(states)
            _, next_values = self.model(next_states)
            returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages

        for _ in range(self.epochs):
            action_probs, current_values = self.model(states)
            dist = torch.distributions.Categorical(action_probs)
            cur_log_probs = dist.log_prob(actions)

            ratios = torch.exp(cur_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(current_values.squeeze(), returns)

            entropy_loss = -self.entropy_coef * dist.entropy().mean()

            loss = actor_loss + self.value_coef * value_loss + entropy_loss

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optim.step()

    def best_action(self, state, width, height):
        state = torch.tensor(state_to_tensor(state, width, height), dtype=torch.float32).unsqueeze(0).to(self.device)
        action_probs, _ = self.model(state)
        return torch.argmax(action_probs).item()

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
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []

        while not (done or truncated):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = state_to_tensor(next_state, env.width, env.height)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done or truncated)

            state = next_state
            total_rewards[episode] += reward
            total_done[episode] = done
            total_steps[episode] = env.steps

            if len(states) == batch_size or done or truncated:
                agent.update(states, actions, log_probs, rewards, next_states, dones)
                states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []

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

    env = GridWorld(20, 20, num_obstacles=20, max_steps=1000)
    agent = PPOAgent(2, env.action_space.n)
    train(env, agent, int(500), 128)
    env.print_policy(agent)
