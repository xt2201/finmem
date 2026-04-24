import argparse
import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
            "Missing RL action artifacts required for 5-measure visualization:\n"
            + "\n".join(missing_paths)
        )
    return loaded


def _required_horizon(price: list[float]) -> int:
    return max(0, len(price) - 1)


def _validate_action_horizon(strategy_name: str, price: list[float], actions: list[int]) -> None:
    required = _required_horizon(price)
    if len(actions) != required:
        raise ValueError(
            f"{strategy_name} action/price horizon mismatch: expected {required}, got {len(actions)}"
        )


def reward_list(price: list[float], actions: list[int]) -> list[float]:
    horizon = _required_horizon(price)
    cumulative = 0.0
    rewards = [0.0]
    for i in range(horizon):
        cumulative += actions[i] * np.log(price[i + 1] / price[i])
        rewards.append(cumulative)
    return rewards


def _prices_from_market_data(
    ticker: str,
    start_time: str,
    end_time: str,
    market_data_path: str,
) -> tuple[list[float], list[pd.Timestamp]]:
    with open(market_data_path, "rb") as f:
        env_data = pickle.load(f)

    start_d = date.fromisoformat(start_time)
    end_d = date.fromisoformat(end_time)
    selected_dates = sorted(
        d
        for d in env_data.keys()
        if start_d <= d <= end_d
        and isinstance(env_data[d], dict)
        and ticker in env_data[d].get("price", {})
    )
    if not selected_dates:
        raise ValueError(
            f"No prices for {ticker} in market data {market_data_path} and date range [{start_time}, {end_time}]."
        )

    prices = [float(env_data[d]["price"][ticker]) for d in selected_dates]
    dates = [pd.Timestamp(d) for d in selected_dates]
    return prices, dates


def _prices_from_vnstock(
    ticker: str,
    start_time: str,
    end_time: str,
) -> tuple[list[float], list[pd.Timestamp]]:
    try:
        from vnstock import Vnstock
    except ImportError as exc:
        raise ImportError(
            "vnstock is required for VN market mode. Install with: pip install vnstock"
        ) from exc

    start_d = date.fromisoformat(start_time)
    end_d = date.fromisoformat(end_time)
    fetch_end = (end_d + timedelta(days=1)).isoformat()

    source = (os.environ.get("FINMEM_VNSTOCK_SOURCE") or "VCI").strip().upper()
    data = Vnstock(show_log=False).stock(symbol=ticker, source=source).quote.history(
        start=start_time,
        end=fetch_end,
        interval="1D",
        show_log=False,
    )

    if "time" in data.columns:
        date_series = pd.to_datetime(data["time"], errors="coerce").dt.date
        data = data.loc[(date_series >= start_d) & (date_series <= end_d)]

    if data.empty:
        raise ValueError(f"No VN market data found for {ticker} in [{start_time}, {end_time}]")

    if "close" in data.columns:
        prices = [float(p) for p in data["close"].tolist()]
    elif "Close" in data.columns:
        prices = [float(p) for p in data["Close"].tolist()]
    else:
        raise ValueError("vnstock output missing close price column.")

    if "time" not in data.columns:
        raise ValueError("vnstock output missing time column.")
    dates = pd.to_datetime(data["time"]).tolist()
    return prices, dates


def _load_price_series(
    ticker: str,
    start_time: str,
    end_time: str,
    market_mode: str,
    market_data_path: str | None,
) -> tuple[list[float], list[pd.Timestamp]]:
    if market_data_path:
        return _prices_from_market_data(ticker, start_time, end_time, market_data_path)
    if market_mode == "VN":
        return _prices_from_vnstock(ticker, start_time, end_time)

    end_fetch = (date.fromisoformat(end_time) + timedelta(days=1)).isoformat()
    data = yf.download(ticker, start=start_time, end=end_fetch, progress=False)
    if data.empty:
        raise ValueError(f"No market data found for {ticker} in [{start_time}, {end_time}]")
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
    return [float(p) for p in prices], dates


def plot_cumulative_returns(dates, return_lists, labels, ticker, file_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    for returns, label in zip(return_lists, labels):
        ax.plot(dates[: len(returns)], returns, label=label, linewidth=2.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(f"{ticker} Cumulative Return Comparison (5 Measures)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format="png", dpi=300)
    print(f"Saved figure to: {file_path}")


def main(
    ticker,
    start_time,
    end_time,
    state_dict_path,
    image_save_path,
    market_mode,
    market_data_path,
    actions_output_dir,
    require_five_measures,
):
    prices, dates = _load_price_series(
        ticker=ticker,
        start_time=start_time,
        end_time=end_time,
        market_mode=market_mode,
        market_data_path=market_data_path,
    )

    finmem_actions = load_finmem_actions_from_state(state_dict_path, start_time, end_time)
    rl_actions = _load_rl_actions(
        ticker=ticker,
        actions_output_dir=actions_output_dir,
        require_five_measures=require_five_measures,
    )

    strategy_actions = {
        "FinMem": finmem_actions,
        "Buy & Hold": [1] * _required_horizon(prices),
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
            "Missing required strategies for 5-measure visualization: "
            + ", ".join(missing_required)
        )

    labels = []
    return_lists = []
    for strategy_name in ordered_strategies:
        if strategy_name not in strategy_actions:
            continue
        _validate_action_horizon(strategy_name, prices, strategy_actions[strategy_name])
        labels.append(strategy_name)
        return_lists.append(reward_list(prices, strategy_actions[strategy_name]))

    plot_cumulative_returns(
        dates=dates,
        return_lists=return_lists,
        labels=labels,
        ticker=ticker,
        file_path=image_save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cumulative return for FinMem test run")
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
        help="Allow missing RL artifacts and visualize available strategies only.",
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
        "figures",
        f"{ticker}_5measures.png",
    )
    main(
        ticker=ticker,
        start_time=args.start,
        end_time=args.end,
        state_dict_path=args.state_dict_path,
        image_save_path=save_path,
        market_mode=market_mode,
        market_data_path=market_data_path,
        actions_output_dir=actions_output_dir,
        require_five_measures=args.require_five_measures,
    )
