import matplotlib.pyplot as plt
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from common.multiprocessing_env import SubprocVecEnv

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

NUM_ENVS = 16
ENV_NAME = "CartPole-v0"
MAX_FRAMES = 20000

HIDDEN_DIM = 256
LEARNING_RATE = 3e-4
NUM_STEPS = 5


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        state_value = self.critic(x)
        action_value = self.actor(x)
        dist = Categorical(action_value)
        return dist, state_value


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


if __name__ == '__main__':
    def make_env():
        def _fun():
            return gym.make(ENV_NAME)

        return _fun


    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = gym.make(ENV_NAME)

    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.n

    model = ActorCritic(state_dim, action_dim, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    # 返回所有环境的状态
    state = envs.reset()
    frame_idx = 0

    while frame_idx < MAX_FRAMES:

        log_probs = []
        state_values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(NUM_STEPS):
            state = torch.FloatTensor(state).to(DEVICE)
            # 所有环境的动作分布(16, 2), 所有环境的状态价值(16,1)
            action_dist, state_value = model(state)

            action = action_dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            # (16,)
            log_prob = action_dist.log_prob(action)
            entropy += action_dist.entropy().mean()

            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(DEVICE))
            # 如果done, mask之后为0, 不用参与reward计算
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(DEVICE))

            state = next_state
            frame_idx += 1

        # 测试10次
        if frame_idx % 1000 == 0:
            print('frame idx:', frame_idx, 'Evaluation Average Reward:',
                  np.mean([test_env() for _ in range(10)]))

        # 训练
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        # [(16, 1),...,(16,1)], 5 steps
        # 进行拼接或平均
        log_probs = torch.cat(log_probs)  # (80, 1)
        returns = torch.cat(returns).detach() # (80, 1)
        state_values = torch.cat(state_values)  # (80, 1)

        advantage = returns - state_values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
