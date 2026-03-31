import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import wilcoxon


def get_price(start, end, ticker):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if "Adj Close" in df.columns:
        return df["Adj Close"].to_numpy()
    if "Close" in df.columns:
        return df["Close"].to_numpy()
    return df.iloc[:, 0].to_numpy()


def get_action(start, end, file_path, col):
    df = pd.read_csv(file_path)
    df[col[0]] = pd.to_datetime(df[col[0]])
    actions = df.loc[
        (df[col[0]] >= pd.to_datetime(start)) & (df[col[0]] <= pd.to_datetime(end)),
        col[1],
    ].to_numpy()
    return actions


def calculate_cumulative_rewards(price, actions):
    horizon = min(len(price) - 1, len(actions))
    reward = 0.0
    reward_list = []
    for i in range(horizon):
        reward += actions[i] * np.log(price[i + 1] / price[i])
        reward_list.append(reward)
    return reward_list


def parse_model_specs(model_specs):
    # Format: Name:path:date_col:action_col
    parsed = {}
    for item in model_specs:
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --model-csv format: '{item}'. Expected Name:path:date_col:action_col"
            )
        model_name, csv_path, date_col, action_col = parts
        parsed[model_name] = (csv_path, [date_col, action_col])
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Run pairwise Wilcoxon tests on model cumulative rewards")
    parser.add_argument("--ticker", default=os.environ.get("FINMEM_TRADING_SYMBOL", "TSLA"))
    parser.add_argument("--start", default=os.environ.get("FINMEM_EVAL_START", "2022-06-25"))
    parser.add_argument("--end", default=os.environ.get("FINMEM_EVAL_END", "2023-04-25"))
    parser.add_argument(
        "--model-csv",
        action="append",
        default=[],
        help="Repeatable. Format: Name:path:date_col:action_col",
    )
    args = parser.parse_args()

    if len(args.model_csv) < 2:
        raise ValueError("Provide at least two --model-csv entries for Wilcoxon comparison.")

    ticker = args.ticker.upper()
    file_paths = parse_model_specs(args.model_csv)

    price = get_price(args.start, args.end, ticker)
    results = {}
    for model, (path, cols) in file_paths.items():
        actions = get_action(args.start, args.end, path, cols)
        results[model] = calculate_cumulative_rewards(price, actions)

    print(f"Wilcoxon Tests for {ticker} from {args.start} to {args.end}")
    model_keys = list(results.keys())
    for i in range(len(model_keys)):
        for j in range(i + 1, len(model_keys)):
            model1 = model_keys[i]
            model2 = model_keys[j]
            rewards1 = results[model1]
            rewards2 = results[model2]
            statistic, pvalue = wilcoxon(rewards1, rewards2)
            print(
                f"Wilcoxon Test between {model1} and {model2} - "
                f"Statistic: {statistic}, P-Value: {pvalue}"
            )


if __name__ == "__main__":
    main()