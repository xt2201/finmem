"""Test 05: Portfolio (position tracking, feedback, momentum)
Validates:
- Price update and day counting
- Action recording and share holding
- Feedback computation (7-day P&L window)
- Momentum computation (3-day window)
- get_action_df output
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
results = {"test": "test_05_portfolio", "passed": [], "failed": []}


def run():
    from puppy.portfolio import Portfolio

    port = Portfolio(symbol="TSLA", lookback_window_size=7)

    # Simulate 10 days of prices and actions
    prices = [750.0, 755.0, 760.0, 758.0, 762.0, 770.0, 765.0, 780.0, 785.0, 790.0]
    actions = [1, 1, -1, 1, 0, 1, -1, 1, -1, 0]  # direction per day
    base_date = date(2022, 7, 14)

    for i, (p, a) in enumerate(zip(prices, actions)):
        cur_date = base_date + timedelta(days=i)
        port.update_market_info(new_market_price_info=p, cur_date=cur_date)
        port.record_action(action={"direction": a})
        port.update_portfolio_series()

    # 1. Day count
    if port.day_count == 10:
        results["passed"].append("day_count")
    else:
        results["failed"].append({"day_count": f"expected 10, got {port.day_count}"})

    # 2. Holding shares
    expected_shares = sum(actions)
    if port.holding_shares == expected_shares:
        results["passed"].append("holding_shares")
    else:
        results["failed"].append({"holding_shares": f"expected {expected_shares}, got {port.holding_shares}"})

    # 3. Feedback response (7-day window, should be available after > 7 days)
    feedback = port.get_feedback_response()
    results["feedback"] = feedback
    if feedback is not None and "feedback" in feedback and feedback["feedback"] in [-1, 0, 1]:
        results["passed"].append("feedback_response")
    else:
        results["failed"].append({"feedback_response": str(feedback)})

    # 4. Momentum (3-day window)
    momentum = port.get_moment(moment_window=3)
    results["momentum"] = momentum
    if momentum is not None and "moment" in momentum and momentum["moment"] in [-1, 0, 1]:
        results["passed"].append("momentum_response")
    else:
        results["failed"].append({"momentum_response": str(momentum)})

    # 5. Action dataframe
    df = port.get_action_df()
    results["action_df_shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
    if df.shape[0] == 10 and df.shape[1] == 3:
        results["passed"].append("action_df_shape")
    else:
        results["failed"].append({"action_df_shape": f"expected (10,3), got {df.shape}"})

    # 6. No feedback if too few days
    port2 = Portfolio(symbol="TSLA", lookback_window_size=7)
    for i in range(3):
        port2.update_market_info(new_market_price_info=prices[i], cur_date=base_date + timedelta(days=i))
        port2.record_action(action={"direction": 1})
        port2.update_portfolio_series()
    if port2.get_feedback_response() is None:
        results["passed"].append("no_feedback_early")
    else:
        results["failed"].append({"no_feedback_early": "expected None"})

    save_and_exit()


def save_and_exit():
    results["summary"] = f"{len(results['passed'])} passed, {len(results['failed'])} failed"
    with open(OUTPUT_DIR / "test_05_portfolio.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
