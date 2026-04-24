from __future__ import annotations

from typing import Dict, List, Tuple

import os
import pickle
import random
from datetime import date

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import torch


def load_env_data(path: str) -> Dict[date, Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_price_series(
    env_data: Dict[date, Dict],
    symbol: str,
    start: date,
    end: date,
) -> Tuple[List[date], np.ndarray]:
    dates = [d for d in env_data.keys() if start <= d <= end]
    dates = sorted(dates)
    prices = [float(env_data[d]["price"][symbol]) for d in dates]
    return dates, np.asarray(prices, dtype=np.float32)


def actions_to_cum_returns(prices: List[float], actions: List[int]) -> List[float]:
    horizon = min(len(actions), len(prices) - 1)
    cumulative = 0.0
    rewards = [0.0]
    for i in range(horizon):
        cumulative += actions[i] * float(np.log(prices[i + 1] / prices[i]))
        rewards.append(cumulative)
    return rewards


def action_indices_to_positions(action_indices: List[int]) -> List[int]:
    mapping = np.array([-1, 0, 1], dtype=np.int32)
    positions = []
    for idx in action_indices:
        if idx < 0 or idx >= len(mapping):
            raise ValueError(f"Action index out of range: {idx}")
        positions.append(int(mapping[idx]))
    return positions


def align_actions_to_price_horizon(
    actions: List[int],
    prices_len: int,
    window: int,
) -> List[int]:
    """Align RL positions to full price transition horizon.

    TradingEnv emits actions only after warmup features are available (offset by
    ``window``). This helper pads leading/trailing transitions with flat position
    (0) so plotted RL series spans the full test date range.
    """
    horizon = max(0, prices_len - 1)
    aligned = [0] * horizon
    if horizon == 0:
        return aligned

    start_idx = max(0, int(window))
    for offset, action in enumerate(actions):
        idx = start_idx + offset
        if idx >= horizon:
            break
        aligned[idx] = int(action)
    return aligned


def load_finmem_actions_aligned(
    state_dict_path: str,
    start: str,
    end: str,
    date_series: List[date],
) -> List[int]:
    with open(state_dict_path, "rb") as f:
        state = pickle.load(f)
    action_df = state["portfolio"].get_action_df()
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    action_map = {
        row["date"]: int(row["direction"])
        for row in action_df.iter_rows(named=True)
        if start_d <= row["date"] < end_d
    }
    return [action_map.get(d, 0) for d in date_series[:-1]]


def plot_cumulative_returns(
    dates: List[date],
    return_lists: List[List[float]],
    labels: List[str],
    ticker: str,
    file_path: str,
) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9.6, 5.37))

    color_map = {
        "B&H": "tab:gray",
        "Buy & Hold": "tab:gray",
        "FinMem": "tab:red",
        "GA": "tab:green",
        "FinGPT": "tab:blue",
        "A2C": "tab:pink",
        "PPO": "tab:orange",
        "DQN": "tab:purple",
    }

    style_map = {
        "B&H": "-.",
        "Buy & Hold": "-.",
        "FinMem": "-",
        "GA": "-",
        "FinGPT": "-",
        "A2C": "-",
        "PPO": "-",
        "DQN": "-",
    }

    for returns, label in zip(return_lists, labels):
        ax.plot(
            dates[: len(returns)],
            returns,
            label=label,
            linewidth=2.0,
            color=color_map.get(label, None),
            linestyle=style_map.get(label, "-"),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"{ticker}")
    ax.legend(frameon=True, loc="lower left")
    ax.grid(True, alpha=0.35)

    if dates:
        left = dates[0].replace(day=1)
        if dates[-1].month == 12:
            right = date(dates[-1].year + 1, 1, 1)
        else:
            right = date(dates[-1].year, dates[-1].month + 1, 1)
        ax.set_xlim(left, right)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format="png", dpi=300)
    print(f"Saved figure to: {file_path}")
