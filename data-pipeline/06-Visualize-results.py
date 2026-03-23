import argparse
import os
import pickle
import sys
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_finmem_actions_from_state(state_dict_path: str, start: str, end: str) -> list[int]:
    with open(state_dict_path, "rb") as f:
        state = pickle.load(f)
    action_df = state["portfolio"].get_action_df()
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    return [
        int(row["direction"])
        for row in action_df.iter_rows(named=True)
        if start_d <= row["date"] < end_d
    ]


def reward_list(price: list[float], actions: list[int]) -> list[float]:
    horizon = min(len(actions), len(price) - 1)
    cumulative = 0.0
    rewards = [0.0]
    for i in range(horizon):
        cumulative += actions[i] * np.log(price[i + 1] / price[i])
        rewards.append(cumulative)
    return rewards


def plot_cumulative_returns(dates, return_lists, labels, ticker, file_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    for returns, label in zip(return_lists, labels):
        ax.plot(dates[: len(returns)], returns, label=label, linewidth=2.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"{ticker} Cumulative Return (FinMem vs Buy & Hold)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format="png", dpi=300)
    print(f"Saved figure to: {file_path}")


def main(ticker, start_time, end_time, state_dict_path, image_save_path):
    data = yf.download(ticker, start=start_time, end=end_time, progress=False)
    if data.empty:
        raise ValueError(f"No market data found for {ticker} in [{start_time}, {end_time})")
    if isinstance(data.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in data.columns:
            prices = data[("Adj Close", ticker)].tolist()
        elif ("Close", ticker) in data.columns:
            prices = data[("Close", ticker)].tolist()
        else:
            prices = data.iloc[:, 0].tolist()
    else:
        if "Adj Close" in data.columns:
            prices = data["Adj Close"].tolist()
        elif "Close" in data.columns:
            prices = data["Close"].tolist()
        else:
            prices = data.iloc[:, 0].tolist()
    dates = pd.to_datetime(data.index).tolist()

    finmem_actions = load_finmem_actions_from_state(state_dict_path, start_time, end_time)
    buy_hold_actions = [1] * max(0, len(prices) - 1)

    finmem_returns = reward_list(prices, finmem_actions)
    buy_hold_returns = reward_list(prices, buy_hold_actions)

    plot_cumulative_returns(
        dates=dates,
        return_lists=[buy_hold_returns, finmem_returns],
        labels=["Buy & Hold", "FinMem"],
        ticker=ticker,
        file_path=image_save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cumulative return for FinMem test run")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--start", default="2022-10-06")
    parser.add_argument("--end", default="2023-04-10")
    parser.add_argument(
        "--state-dict-path",
        default="data/09_results_minilm/agent_1/state_dict.pkl",
        help="Path to FinMem test output agent state_dict.pkl",
    )
    parser.add_argument(
        "--save-path",
        default="figures/TSLA_finmem_vs_buyhold.png",
    )
    args = parser.parse_args()
    main(args.ticker, args.start, args.end, args.state_dict_path, args.save_path)
