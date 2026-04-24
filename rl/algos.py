from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: Tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: Tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.base = MLP(input_dim, hidden[-1], hidden=hidden[:-1] if len(hidden) > 1 else (hidden[0],))
        self.policy = nn.Linear(hidden[-1], action_dim)
        self.value = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.base(x)
        return self.policy(z), self.value(z).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.buffer[i] for i in idx))
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 20000
    target_update: int = 200


class DQNAgent:
    def __init__(self, obs_size: int, action_size: int, config: DQNConfig) -> None:
        self.q_net = MLP(obs_size, action_size)
        self.target_net = MLP(obs_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(capacity=config.buffer_size)
        self.config = config
        self.step_count = 0
        self.action_size = action_size

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_size))
        with torch.no_grad():
            q = self.q_net(torch.from_numpy(obs).unsqueeze(0))
            return int(torch.argmax(q, dim=-1).item())

    def learn(self) -> float:
        if len(self.buffer) < self.config.batch_size:
            return 0.0
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config.batch_size)
        obs_t = torch.from_numpy(obs)
        actions_t = torch.from_numpy(actions).unsqueeze(1)
        rewards_t = torch.from_numpy(rewards)
        next_obs_t = torch.from_numpy(next_obs)
        dones_t = torch.from_numpy(dones)

        q_values = self.q_net(obs_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(dim=1)[0]
            target = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)
        loss = torch.mean((q_values - target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.config.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def predict(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            q = self.q_net(torch.from_numpy(obs).unsqueeze(0))
            return int(torch.argmax(q, dim=-1).item())


@dataclass
class A2CConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01


class A2CAgent:
    def __init__(self, obs_size: int, action_size: int, config: A2CConfig) -> None:
        self.net = ActorCritic(obs_size, action_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.config = config
        self.action_size = action_size

    def act(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0), entropy.squeeze(0)

    def predict(self, obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        logits, _ = self.net(obs_t)
        return int(torch.argmax(logits, dim=-1).item())


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 64


class PPOAgent:
    def __init__(self, obs_size: int, action_size: int, config: PPOConfig) -> None:
        self.net = ActorCritic(obs_size, action_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.config = config

    def act(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def predict(self, obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        logits, _ = self.net(obs_t)
        return int(torch.argmax(logits, dim=-1).item())
