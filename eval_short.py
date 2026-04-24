import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sys.path.append('data-pipeline')

from importlib.machinery import SourceFileLoader
metrics = SourceFileLoader("metrics", "data-pipeline/07-metrics.py").load_module()

import argparse
parser = argparse.ArgumentParser(description="Evaluate simulation results")
parser.add_argument("--pkl-path", type=str, default="data/10_short_eval/test_output/agent_1/state_dict.pkl", help="Path to state_dict.pkl")
args = parser.parse_args()

# Load specific mini-test checkpoint output
pkl_path = args.pkl_path
try:
    with open(pkl_path, 'rb') as f:
        state = pickle.load(f)
except FileNotFoundError:
    print(f"Could not find short eval file at {pkl_path}")
    sys.exit(1)

portfolio = state['portfolio']
prices_array = portfolio.market_price_series
actions_dict = portfolio.action_series
dates = portfolio.date_series

# Price list handles array-like nicely
price_list = prices_array.tolist() if hasattr(prices_array, 'tolist') else list(prices_array)

# For day i, the action applied to the return from day i to i+1 is action_list[i]
action_list = [actions_dict.get(d, 0) for d in dates[:-1]]

# 1. FinMem Metrics
finmem_daily_rewards = metrics.daily_reward(price_list, action_list)
finmem_std_dev = metrics.standard_deviation(finmem_daily_rewards) if len(finmem_daily_rewards) > 1 else 0
finmem_total_reward = metrics.total_reward(price_list, action_list)
finmem_ann_vol = metrics.annualized_volatility(finmem_std_dev)
try:
    finmem_sharpe = metrics.calculate_sharpe_ratio(
        finmem_total_reward,
        0.0,
        finmem_ann_vol,
        max(0, len(price_list) - 1),
    ) if finmem_ann_vol > 0 else 0
except:
    finmem_sharpe = 0
finmem_mdd = metrics.calculate_max_drawdown(finmem_daily_rewards)

# 2. Buy & Hold Metrics
bh_action_list = [1] * max(0, len(price_list) - 1)
bh_daily_rewards = metrics.daily_reward(price_list, bh_action_list)
bh_std_dev = metrics.standard_deviation(bh_daily_rewards) if len(bh_daily_rewards) > 1 else 0
bh_total_reward = metrics.total_reward(price_list, bh_action_list)
bh_ann_vol = metrics.annualized_volatility(bh_std_dev)
try:
    bh_sharpe = metrics.calculate_sharpe_ratio(
        bh_total_reward,
        0.0,
        bh_ann_vol,
        max(0, len(price_list) - 1),
    ) if bh_ann_vol > 0 else 0
except:
    bh_sharpe = 0
bh_mdd = metrics.calculate_max_drawdown(bh_daily_rewards)

print("=== SHORT PERIOD EVALUATION RESULTS ===")
print(f"Period: {dates[0]} to {dates[-1]}")
print(f"Trading Days: {len(dates)}\n")

print("--- Daily Logs ---")
for i, d in enumerate(dates[:-1]):
    a = action_list[i]
    act_str = "BUY" if a == 1 else ("SELL" if a == -1 else "HOLD")
    print(f"{d}: Price={price_list[i]:.2f} | Action={act_str}")
if dates:
    print(f"{dates[-1]}: Price={price_list[-1]:.2f} | Action=N/A (terminal day)")

print("\n--- [FinMem] ---")
print(f"Total Cumulative Reward : {finmem_total_reward:.4f} ({(np.exp(finmem_total_reward)-1)*100:.2f}%)")
print(f"Annualized Volatility   : {finmem_ann_vol:.4f}")
print(f"Sharpe Ratio            : {finmem_sharpe:.4f}")
print(f"Max Drawdown            : {finmem_mdd:.4f}")

print("\n--- [Buy & Hold] ---")
print(f"Total Cumulative Reward : {bh_total_reward:.4f} ({(np.exp(bh_total_reward)-1)*100:.2f}%)")
print(f"Annualized Volatility   : {bh_ann_vol:.4f}")
print(f"Sharpe Ratio            : {bh_sharpe:.4f}")
print(f"Max Drawdown            : {bh_mdd:.4f}")

# 3. Create Plot
import os
os.makedirs('figures', exist_ok=True)

# Cumulative return lists start at 0
ret_f = [0]
c_f = 0
for r in finmem_daily_rewards:
    c_f += r
    ret_f.append(c_f)

ret_bh = [0]
c_bh = 0
for r in bh_daily_rewards:
    c_bh += r
    ret_bh.append(c_bh)

# Format dates for x-axis
dates_str = [d.strftime("%m-%d") for d in dates]

plt.figure(figsize=(10, 6))
plt.plot(dates_str, ret_bh, label='Buy & Hold', color='red', linestyle='--', linewidth=2)
plt.plot(dates_str, ret_f, label='FinMem', color='blue', linewidth=2)

plt.title('Cumulative Returns: FinMem vs Buy & Hold (Short Test Period)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Log)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

img_path = 'figures/short_eval_plot.png'
plt.savefig(img_path, dpi=300)
print(f"\n[+] Visualization saved to {img_path}")