from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .algos import DQNAgent, A2CAgent, PPOAgent
from .env import TradingEnv


@dataclass
class TrainResult:
    actions: List[int]
    rewards: List[float]
    losses: List[float]


def train_dqn(
    env: TradingEnv,
    agent: DQNAgent,
    episodes: int = 20,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.98,
) -> TrainResult:
    losses: List[float] = []
    rewards: List[float] = []
    actions: List[int] = []
    epsilon = epsilon_start

    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, epsilon)
            next_obs, reward, done, _ = env.step(action)
            agent.buffer.add(obs, action, reward, next_obs, done)
            loss = agent.learn()
            obs = next_obs
            losses.append(loss)
            rewards.append(reward)
            actions.append(action)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return TrainResult(actions=actions, rewards=rewards, losses=losses)


def train_a2c(env: TradingEnv, agent: A2CAgent, episodes: int = 20) -> TrainResult:
    losses: List[float] = []
    rewards: List[float] = []
    actions: List[int] = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, log_prob, value, entropy = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            _, next_value = agent.net(torch.from_numpy(next_obs).unsqueeze(0))
            td_target = reward + agent.config.gamma * next_value.squeeze(0) * (1.0 - float(done))
            advantage = td_target.detach() - value
            policy_loss = -(log_prob * advantage.detach())
            value_loss = agent.config.value_coeff * (advantage ** 2)
            entropy_loss = -agent.config.entropy_coeff * entropy
            loss = policy_loss + value_loss + entropy_loss

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            obs = next_obs
            losses.append(float(loss.item()))
            rewards.append(float(reward))
            actions.append(action)

    return TrainResult(actions=actions, rewards=rewards, losses=losses)


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train_ppo(env: TradingEnv, agent: PPOAgent, episodes: int = 20) -> TrainResult:
    losses: List[float] = []
    rewards: List[float] = []
    actions: List[int] = []

    for _ in range(episodes):
        obs_list: List[np.ndarray] = []
        action_list: List[int] = []
        reward_list: List[float] = []
        done_list: List[float] = []
        logprob_list: List[float] = []
        value_list: List[float] = []

        obs = env.reset()
        done = False
        while not done:
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(float(reward))
            done_list.append(float(done))
            logprob_list.append(float(log_prob.item()))
            value_list.append(float(value.item()))
            rewards.append(float(reward))
            actions.append(action)
            obs = next_obs

        advantages, returns = _compute_gae(
            np.asarray(reward_list, dtype=np.float32),
            np.asarray(value_list, dtype=np.float32),
            np.asarray(done_list, dtype=np.float32),
            agent.config.gamma,
            agent.config.lam,
        )

        obs_t = torch.from_numpy(np.asarray(obs_list, dtype=np.float32))
        actions_t = torch.from_numpy(np.asarray(action_list, dtype=np.int64))
        old_log_probs_t = torch.from_numpy(np.asarray(logprob_list, dtype=np.float32))
        adv_t = torch.from_numpy(advantages.astype(np.float32))
        ret_t = torch.from_numpy(returns.astype(np.float32))

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_steps = len(obs_list)
        batch_size = agent.config.minibatch_size
        idxs = np.arange(total_steps)
        for _ in range(agent.config.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log = old_log_probs_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                log_probs, values, entropy = agent.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log)
                clipped = torch.clamp(ratio, 1.0 - agent.config.clip_eps, 1.0 + agent.config.clip_eps)
                policy_loss = -(torch.min(ratio * mb_adv, clipped * mb_adv)).mean()
                value_loss = agent.config.value_coeff * (mb_ret - values).pow(2).mean()
                entropy_loss = -agent.config.entropy_coeff * entropy.mean()
                loss = policy_loss + value_loss + entropy_loss

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
                losses.append(float(loss.item()))

    return TrainResult(actions=actions, rewards=rewards, losses=losses)


def evaluate_agent(env: TradingEnv, agent, deterministic: bool = True) -> List[int]:
    obs = env.reset()
    done = False
    actions: List[int] = []
    while not done:
        if deterministic:
            action = agent.predict(obs)
        else:
            if hasattr(agent, "act"):
                action = agent.act(obs)[0]
            else:
                action = agent.predict(obs)
        next_obs, _, done, _ = env.step(action)
        actions.append(int(action))
        obs = next_obs
    return actions
