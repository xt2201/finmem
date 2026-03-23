import argparse
import os
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_price(start: str, end: str, ticker: str) -> list[float]:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No price data downloaded for {ticker} from {start} to {end}")
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            return df[("Adj Close", ticker)].tolist()
        if ("Close", ticker) in df.columns:
            return df[("Close", ticker)].tolist()
        return df.iloc[:, 0].tolist()
    if "Adj Close" in df.columns:
        return df["Adj Close"].tolist()
    if "Close" in df.columns:
        return df["Close"].tolist()
    return df.iloc[:, 0].tolist()


def load_finmem_actions_from_state(state_dict_path: str, start: str, end: str) -> list[int]:
    with open(state_dict_path, "rb") as f:
        state = pickle.load(f)
    portfolio = state["portfolio"]
    # Polars DataFrame with columns: date, symbol, direction
    action_df = portfolio.get_action_df()
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    filtered = [
        int(row["direction"])
        for row in action_df.iter_rows(named=True)
        if start_d <= row["date"] < end_d
    ]
    return filtered


def daily_reward(price: list[float], actions: list[int]) -> list[float]:
    horizon = min(len(actions), len(price) - 1)
    reward = []
    for i in range(horizon):
        reward.append(actions[i] * np.log(price[i + 1] / price[i]))
    return reward


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def total_reward(price: list[float], actions: list[int]) -> float:
    horizon = min(len(actions), len(price) - 1)
    reward = 0.0
    for i in range(horizon):
        reward += actions[i] * np.log(price[i + 1] / price[i])
    return reward


def annualized_volatility(daily_std_dev: float, trading_days: int = 252) -> float:
    return daily_std_dev * (trading_days ** 0.5)


def calculate_sharpe_ratio(rp: float, rf: float, sigma_p: float, n_price: int) -> float:
    if sigma_p == 0 or n_price == 0:
        return 0.0
    rp_annual = rp / (n_price / 252)
    return (rp_annual - rf) / sigma_p


def calculate_max_drawdown(daily_returns: list[float]) -> float:
    cumulative_returns = [1.0]
    for r in daily_returns:
        cumulative_returns.append(cumulative_returns[-1] * (1 + r))
    peak = cumulative_returns[0]
    max_drawdown = 0.0
    for r in cumulative_returns:
        if r > peak:
            peak = r
        drawdown = (peak - r) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def calculate_metrics(price: list[float], actions: list[int]) -> tuple[float, float, float, float, float]:
    daily_rw = daily_reward(price, actions)
    std_dev_r = standard_deviation(daily_rw)
    ann_vol = annualized_volatility(std_dev_r)
    cum_return = total_reward(price, actions)
    sharpe_ratio = calculate_sharpe_ratio(cum_return, 0.0, ann_vol, len(price))
    max_dd = calculate_max_drawdown(daily_rw)
    return cum_return, sharpe_ratio, std_dev_r, ann_vol, max_dd


def main(ticker: str, start: str, end: str, state_dict_path: str, save_path: str) -> None:
    price = get_price(start, end, ticker)
    actions = load_finmem_actions_from_state(state_dict_path, start, end)

    # Align to returns horizon (len(price) - 1)
    actions = actions[: max(0, len(price) - 1)]
    buy_hold_actions = [1] * max(0, len(price) - 1)

    metrics = [
        "Cumulative Return",
        "Sharpe Ratio",
        "Standard Deviation",
        "Annualized Volatility",
        "Max Drawdown",
    ]
    results = {
        "Buy & Hold": calculate_metrics(price, buy_hold_actions),
        "FinMem": calculate_metrics(price, actions),
    }

    df_results = pd.DataFrame(results, index=metrics)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_results.to_csv(save_path)
    print(df_results)
    print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for FinMem test run")
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
        default="data/07_test_model_output/TSLA_metrics_finmem.csv",
    )
    args = parser.parse_args()
    main(args.ticker, args.start, args.end, args.state_dict_path, args.save_path)
