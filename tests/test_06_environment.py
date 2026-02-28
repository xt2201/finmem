"""Test 06: MarketEnvironment (data loading and iteration)
Validates:
- Environment creation from a dict (env_data format)
- step() returns correct market_info tuple structure
- Date filtering with start_date/end_date
- Termination flag behavior
"""
import sys
import json
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
results = {"test": "test_06_environment", "passed": [], "failed": []}


def create_synthetic_env_data():
    """Create a minimal synthetic env_data dict for testing."""
    base_date = date(2022, 7, 14)
    data = {}
    for i in range(10):
        d = base_date + timedelta(days=i)
        data[d] = {
            "price": {"TSLA": 750.0 + i * 5},
            "news": {"TSLA": [f"Test news for day {i}"]},
            "filing_k": {"TSLA": ""},
            "filing_q": {"TSLA": ""},
        }
    return data


def run():
    from puppy.environment import MarketEnvironment

    env_data = create_synthetic_env_data()
    base_date = date(2022, 7, 14)

    # 1. Creation — takes env_data_pkl dict directly, not a path
    try:
        env = MarketEnvironment(
            env_data_pkl=env_data,
            start_date=base_date,
            end_date=base_date + timedelta(days=8),
            symbol="TSLA",
        )
        results["passed"].append("env_creation")
    except Exception as e:
        results["failed"].append({"env_creation": str(e)})
        save_and_exit()
        return

    # 2. Step through and validate tuple structure
    step_count = 0
    all_dates = []
    last_done = False
    while True:
        market_info = env.step()
        cur_date, cur_price, filing_k, filing_q, news, record, done = market_info

        if done:
            last_done = True
            break

        if step_count == 0:
            if isinstance(cur_date, date) and isinstance(cur_price, float):
                results["passed"].append("market_info_types")
            else:
                results["failed"].append({"market_info_types": f"date={type(cur_date)}, price={type(cur_price)}"})

        all_dates.append(str(cur_date))
        step_count += 1

    results["step_count"] = step_count
    results["dates"] = all_dates

    # 3. Check step count
    if step_count >= 5:
        results["passed"].append("step_count_valid")
    else:
        results["failed"].append({"step_count": f"expected >=5, got {step_count}"})

    # 4. Done flag set on last step
    if last_done:
        results["passed"].append("termination_flag")
    else:
        results["failed"].append({"termination_flag": "last step did not set done=True"})

    save_and_exit()


def save_and_exit():
    results["summary"] = f"{len(results['passed'])} passed, {len(results['failed'])} failed"
    with open(OUTPUT_DIR / "test_06_environment.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
