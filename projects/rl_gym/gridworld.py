from enum import Enum
import gymnasium as gym
import numpy as np
import random

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class GridWorld(gym.Env):
    def __init__(self, width, height, num_obstacles=None, max_steps=1e9):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles if num_obstacles is not None else width * height // 16

        self.observation_space = gym.spaces.Discrete(width * height)
        self.action_space = gym.spaces.Discrete(4) # up: 0, right: 1, down: 2, left: 3

        self.start_state = 0
        self.state = self.start_state
        self.goal_state = width * height - 1

        self.max_steps = max_steps
        self.steps = 0

        self.obstacles = set(random.sample(range(1, width * height - 1), self.num_obstacles)) # exclude start and goal states
        # self.obstacles = set()
        # while len(self.obstacles) < self.num_obstacles:
        #     obstacle = random.randint(1, self.width * self.height - 2) # exclude goal state
        #     if obstacle not in self.obstacles:
        #         self.obstacles.add(obstacle)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.start_state
        self.steps = 0
        return self.state, {}

    def step(self, action):
        action = Action(action)
        x = self.state % self.width
        y = self.state // self.width

        if action == Action.UP:
            y = max(0, y - 1)
        elif action == Action.RIGHT:
            x = min(self.width - 1, x + 1)
        elif action == Action.DOWN:
            y = min(self.height - 1, y + 1)
        elif action == Action.LEFT:
            x = max(0, x - 1)

        new_state = y * self.width + x

        reward = -0.1

        if new_state == self.state:
            reward = -1.0

        if new_state in self.obstacles:
            reward = -1.0
            new_state = self.state

        self.state = new_state

        done = self.state == self.goal_state
        if done:
            reward = 100.0

        self.steps += 1
        truncated = self.steps >= self.max_steps

        info = {}

        return self.state, reward, done, truncated, info

    def print_policy(self, agent):
        actions = ["↑", "→", "↓", "←"]
        for y in range(self.height):
            # env
            for x in range(self.width):
                state = y * self.width + x
                if state == self.state:
                    print("A", end=' ')
                elif state in self.obstacles:
                    print("▇", end=' ')
                elif state == self.goal_state:
                    print("★", end=' ')
                else:
                    print(".", end=' ')
            print('', end='\t')

            # policy
            for x in range(self.width):
                state = y * self.width + x
                best_action = agent.best_action(state, self.width, self.height)
                if state in self.obstacles:
                    print("▇", end=' ')
                elif state == self.start_state or state == self.goal_state:
                    print(" ", end=' ')
                else:
                    print(actions[best_action], end=' ')
            print()