import argparse
import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MARKET_MODE = "US"
RL_ALGORITHMS = ("DQN", "A2C", "PPO")


def _resolve_market_mode(raw: str) -> str:
    market = (raw or DEFAULT_MARKET_MODE).strip().upper().replace("-", "_")
    if market in {"US", "USA", "U.S.", "U_S"}:
        return "US"
    if market in {"VN", "VIETNAM", "VIET_NAM", "VNSE"}:
        return "VN"
    raise ValueError(f"Unsupported market mode '{raw}'. Supported values: US, VN.")


def _get_price_from_market_data(
    start: str,
    end: str,
    ticker: str,
    market_data_path: str,
) -> list[float]:
    with open(market_data_path, "rb") as f:
        env_data = pickle.load(f)

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    selected_dates = sorted(
        d
        for d in env_data.keys()
        if start_d <= d <= end_d
        and isinstance(env_data[d], dict)
        and ticker in env_data[d].get("price", {})
    )
    prices = [float(env_data[d]["price"][ticker]) for d in selected_dates]
    if not prices:
        raise ValueError(
            f"No prices for {ticker} in market data {market_data_path} and date range [{start}, {end}]."
        )
    return prices


def _get_price_vnstock(start: str, end: str, ticker: str) -> list[float]:
    try:
        from vnstock import Vnstock
    except ImportError as exc:
        raise ImportError(
            "vnstock is required for VN market mode. Install with: pip install vnstock"
        ) from exc

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    fetch_end = (end_d + timedelta(days=1)).isoformat()

    source = (os.environ.get("FINMEM_VNSTOCK_SOURCE") or "VCI").strip().upper()
    df = Vnstock(show_log=False).stock(symbol=ticker, source=source).quote.history(
        start=start,
        end=fetch_end,
        interval="1D",
        show_log=False,
    )

    if "time" in df.columns:
        date_series = pd.to_datetime(df["time"], errors="coerce").dt.date
        df = df.loc[(date_series >= start_d) & (date_series <= end_d)]

    if df.empty:
        raise ValueError(f"No VN price data downloaded for {ticker} from {start} to {end}")

    if "close" in df.columns:
        prices = df["close"].tolist()
    elif "Close" in df.columns:
        prices = df["Close"].tolist()
    else:
        raise ValueError("vnstock output missing close price column.")

    if not prices:
        raise ValueError(f"No VN close prices found for {ticker} from {start} to {end}")
    return [float(p) for p in prices]


def get_price(
    start: str,
    end: str,
    ticker: str,
    market_mode: str,
    market_data_path: str | None,
) -> list[float]:
    if market_data_path:
        return _get_price_from_market_data(start, end, ticker, market_data_path)

    if market_mode == "VN":
        return _get_price_vnstock(start, end, ticker)

    end_fetch = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    df = yf.download(ticker, start=start, end=end_fetch, progress=False)
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
    action_df = portfolio.get_action_df().sort("date")
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    filtered = [
        int(row["direction"])
        for row in action_df.iter_rows(named=True)
        if start_d <= row["date"] < end_d
    ]
    return filtered


def _load_actions_from_artifact(artifact_path: str) -> list[int]:
    with open(artifact_path, "rb") as f:
        payload = pickle.load(f)

    raw_actions = payload.get("actions") if isinstance(payload, dict) else payload
    if raw_actions is None:
        raise ValueError(f"Missing 'actions' field in artifact: {artifact_path}")

    if isinstance(raw_actions, np.ndarray):
        action_values = raw_actions.tolist()
    elif isinstance(raw_actions, tuple):
        action_values = list(raw_actions)
    elif isinstance(raw_actions, list):
        action_values = raw_actions
    else:
        raise ValueError(f"Unsupported action artifact format in {artifact_path}")

    return [int(v) for v in action_values]


def _load_rl_actions(
    ticker: str,
    actions_output_dir: str,
    require_five_measures: bool,
) -> dict[str, list[int]]:
    loaded = {}
    missing_paths = []

    for algo in RL_ALGORITHMS:
        artifact_path = os.path.join(actions_output_dir, f"{ticker}_actions_{algo.lower()}.pkl")
        if not os.path.exists(artifact_path):
            missing_paths.append(artifact_path)
            continue
        loaded[algo] = _load_actions_from_artifact(artifact_path)

    if require_five_measures and missing_paths:
        raise FileNotFoundError(
            "Missing RL action artifacts required for 5 measures:\n"
            + "\n".join(missing_paths)
        )
    return loaded


def _required_horizon(price: list[float]) -> int:
    return max(0, len(price) - 1)


def _validate_action_horizon(price: list[float], actions: list[int]) -> int:
    required = _required_horizon(price)
    if len(actions) != required:
        raise ValueError(
            f"Action/price horizon mismatch: expected {required} actions for {len(price)} prices, got {len(actions)}."
        )
    return required


def daily_reward(price: list[float], actions: list[int]) -> list[float]:
    horizon = _validate_action_horizon(price, actions)
    return [actions[i] * np.log(price[i + 1] / price[i]) for i in range(horizon)]


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def total_reward(price: list[float], actions: list[int]) -> float:
    horizon = _validate_action_horizon(price, actions)
    reward = 0.0
    for i in range(horizon):
        reward += actions[i] * np.log(price[i + 1] / price[i])
    return reward


def annualized_volatility(daily_std_dev: float, trading_days: int = 252) -> float:
    return daily_std_dev * (trading_days ** 0.5)


def calculate_sharpe_ratio(
    rp: float,
    rf: float,
    sigma_p: float,
    n_periods: int,
    trading_days: int = 252,
) -> float:
    if sigma_p <= 0 or n_periods <= 0:
        return 0.0
    rp_annual = rp * (trading_days / n_periods)
    return (rp_annual - rf) / sigma_p


def calculate_max_drawdown(daily_returns: list[float]) -> float:
    cumulative_returns = [1.0]
    for r in daily_returns:
        # daily_returns are log-returns, so convert with exp for wealth compounding.
        cumulative_returns.append(cumulative_returns[-1] * float(np.exp(r)))
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
    n_periods = _validate_action_horizon(price, actions)
    daily_rw = daily_reward(price, actions)
    std_dev_r = standard_deviation(daily_rw)
    ann_vol = annualized_volatility(std_dev_r)
    cum_return = total_reward(price, actions)
    sharpe_ratio = calculate_sharpe_ratio(cum_return, 0.0, ann_vol, n_periods)
    max_dd = calculate_max_drawdown(daily_rw)
    return cum_return, sharpe_ratio, std_dev_r, ann_vol, max_dd


def main(
    ticker: str,
    start: str,
    end: str,
    state_dict_path: str,
    save_path: str,
    market_mode: str,
    market_data_path: str | None,
    actions_output_dir: str,
    require_five_measures: bool,
) -> None:
    price = get_price(start, end, ticker, market_mode, market_data_path)
    actions = load_finmem_actions_from_state(state_dict_path, start, end)
    rl_actions = _load_rl_actions(
        ticker=ticker,
        actions_output_dir=actions_output_dir,
        require_five_measures=require_five_measures,
    )

    required_horizon = _required_horizon(price)
    if len(actions) != required_horizon:
        raise ValueError(
            (
                f"Loaded {len(actions)} FinMem actions but expected {required_horizon} for "
                f"price horizon [{start}, {end}]. Please re-run test with matching date range."
            )
        )

    strategy_actions = {
        "FinMem": actions,
        "Buy & Hold": [1] * _required_horizon(price),
    }
    for algo in RL_ALGORITHMS:
        if algo in rl_actions:
            strategy_actions[algo] = rl_actions[algo]

    ordered_strategies = ["FinMem", "Buy & Hold", "A2C", "DQN", "PPO"]
    missing_required = [
        name for name in ordered_strategies if require_five_measures and name not in strategy_actions
    ]
    if missing_required:
        raise ValueError(
            "Missing required strategies for 5-measure report: "
            + ", ".join(missing_required)
        )

    metrics = [
        "Cumulative Return",
        "Sharpe Ratio",
        "Standard Deviation",
        "Annualized Volatility",
        "Max Drawdown",
    ]
    results = {}
    for strategy_name in ordered_strategies:
        if strategy_name not in strategy_actions:
            continue
        results[strategy_name] = calculate_metrics(price, strategy_actions[strategy_name])

    df_results = pd.DataFrame(results, index=metrics)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_results.to_csv(save_path)
    print(df_results)
    print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for FinMem test run")
    parser.add_argument("--ticker", default=os.environ.get("FINMEM_TRADING_SYMBOL", "TSLA"))
    parser.add_argument(
        "--market",
        default=os.environ.get("FINMEM_MARKET_MODE") or os.environ.get("FINMEM_MARKET") or DEFAULT_MARKET_MODE,
        help="Market mode: US or VN.",
    )
    parser.add_argument("--start", default=os.environ.get("FINMEM_EVAL_START", "2022-10-06"))
    parser.add_argument("--end", default=os.environ.get("FINMEM_EVAL_END", "2023-04-10"))
    parser.add_argument(
        "--market-data-path",
        default=os.environ.get("FINMEM_MARKET_DATA_PATH", ""),
        help="Optional env_data pickle path used as direct price source.",
    )
    parser.add_argument(
        "--state-dict-path",
        default=os.environ.get("FINMEM_STATE_DICT_PATH", "data/09_results_minilm/agent_1/state_dict.pkl"),
        help="Path to FinMem test output agent state_dict.pkl",
    )
    parser.add_argument(
        "--actions-output-dir",
        default=(
            os.environ.get("FINMEM_RL_ACTIONS_OUTPUT_DIR")
            or os.environ.get("FINMEM_RESULT_PATH")
            or os.path.join("data", "09_results")
        ),
        help="Directory containing RL action artifacts: <TICKER>_actions_{dqn|a2c|ppo}.pkl",
    )
    parser.add_argument(
        "--require-five-measures",
        dest="require_five_measures",
        action="store_true",
        default=True,
        help="Require all 5 strategies (FinMem, Buy & Hold, A2C, DQN, PPO). Enabled by default.",
    )
    parser.add_argument(
        "--allow-missing-rl",
        dest="require_five_measures",
        action="store_false",
        help="Allow missing RL artifacts and report available strategies only.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()
    market_mode = _resolve_market_mode(args.market)
    market_data_path = args.market_data_path.strip() or None
    actions_output_dir = args.actions_output_dir
    save_path = args.save_path or os.path.join(
        "data",
        "07_test_model_output",
        f"{ticker}_metrics_5measures.csv",
    )
    main(
        ticker=ticker,
        start=args.start,
        end=args.end,
        state_dict_path=args.state_dict_path,
        save_path=save_path,
        market_mode=market_mode,
        market_data_path=market_data_path,
        actions_output_dir=actions_output_dir,
        require_five_measures=args.require_five_measures,
    )
