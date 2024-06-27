import numpy as np

from gridworld import GridWorld

class QAgent:
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.95, eps=0.1):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.action_size = action_size

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def best_action(self, state, w, h):
        return np.argmax(agent.q_table[state])

    def update(self, state, action, reward, next_state, done):
        cur_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target_q - cur_q)

def train(env, agent, episodes):
    total_rewards = np.zeros(episodes)
    total_steps = np.zeros(episodes)
    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_rewards[episode] += reward
            total_steps[episode] = env.steps

            if truncated:
                break

        if episode % 100 == 0:
            avg_reward = np.mean(total_rewards[max(0, episode - 100):episode + 1])
            avg_steps = np.mean(total_steps[max(0, episode - 100):episode + 1])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.3f}, Steps: {avg_steps}")

seed=4
import torch
import random
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

env = GridWorld(10, 10, num_obstacles=8, max_steps=500)
agent = QAgent(env.observation_space.n, env.action_space.n)
train(env, agent, int(1e3))
env.print_policy(agent)
