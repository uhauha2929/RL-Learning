# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 20:04
# @Author  : uhauha2929
from typing import Union

import gym
import torch
import torch.nn.functional as func
import numpy as np

EPISODE = 2000  # Episode limitation
STEP = 1000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode
GAMMA = 0.95  # discount factor
LEARNING_RATE = 0.005
HIDDEN_DIM = 128


class Policy(torch.nn.Module):
    """
    softmax network, 用来预测最优动作(策略)
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state_input: torch.Tensor):
        return self.feedforward(state_input)


class PolicyGradient(object):

    def __init__(self, env):
        self.env = env
        # 状态空间的维度, 4
        self.state_dim = env.observation_space.shape[0]
        # 动作空间的维度, 2
        self.action_dim = env.action_space.n
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.policy = Policy(self.state_dim, HIDDEN_DIM, self.action_dim)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

    def choose_action(self, observation: list):
        # (batch_size, state_dim), 这里默认每次输入一条状态
        observation = torch.Tensor(observation).view(-1, self.state_dim)
        with torch.no_grad():
            actions_prob = self.policy(observation).flatten().numpy()
            # 以预测的actions的概率随机选择
            action = np.random.choice(range(self.action_dim), p=actions_prob)
        return action

    def store_transition(self, state: list, action: int, reward: float):
        # 分别存放一个Episode的所有观察到的状态, 采取的动作, 和回报
        self.episode_observations.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def train(self):
        # 从后向前计算折扣的回报期望来代替每个时间位置t的状态价值v_t
        discounted_rewards = np.zeros_like(self.episode_rewards)
        running_add = 0
        for t in reversed(range(0, len(self.episode_rewards))):
            running_add = running_add * GAMMA + self.episode_rewards[t]
            discounted_rewards[t] = running_add

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        discounted_rewards = torch.Tensor(discounted_rewards)

        self.optimizer.zero_grad()
        # train on one episode
        observation = torch.FloatTensor(self.episode_observations).view(-1, self.state_dim)
        actions_prob = self.policy(observation)
        actions = torch.LongTensor(self.episode_actions)
        # 计算loss, reduction='none', 不求和或平均
        loss = func.nll_loss(actions_prob, actions, reduction='none')
        loss = torch.mean(loss * discounted_rewards)
        loss.backward()
        self.optimizer.step()

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []  # empty episode data


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = PolicyGradient(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()  # state为一个四维向量特征表示
        # Train
        for step in range(STEP):
            action = agent.choose_action(state)  # e-greedy action for train, 0或1的值
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            if done:
                agent.train()
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
