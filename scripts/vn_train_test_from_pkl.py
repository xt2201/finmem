#!/usr/bin/env python3
"""Derive FinMem TRAIN/TEST date env values from a VN (or any) market pickle by calendar month.

Example:
  .venv/bin/python scripts/vn_train_test_from_pkl.py \\
    data/03_model_input/bid.pkl --train-month 2025-04 --test-month 2025-05 --export-bash
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date
from typing import List, Tuple


def _filter_month(days: List[date], y: int, m: int) -> List[date]:
    return [d for d in days if d.year == y and d.month == m]


def _parse_ym(ym: str) -> Tuple[int, int]:
    parts = ym.strip().split("-")
    if len(parts) != 2:
        raise ValueError("Expected YYYY-MM")
    y, m = int(parts[0]), int(parts[1])
    if not (1 <= m <= 12):
        raise ValueError("Month must be 1-12")
    return y, m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pkl", help="Path to data/03_model_input/<symbol>.pkl")
    p.add_argument(
        "--train-month",
        required=True,
        help="First month of training window (e.g. 2025-04). Uses all trading days in that month present in the pickle.",
    )
    p.add_argument(
        "--test-month",
        required=True,
        help="Test month (e.g. 2025-05).",
    )
    p.add_argument(
        "--export-bash",
        action="store_true",
        help="Print `export TRAIN_START_DATE=...` lines for bash",
    )
    args = p.parse_args()

    with open(args.pkl, "rb") as f:
        data: dict = pickle.load(f)

    days = sorted(data.keys())
    if not days:
        print("Error: empty pickle", file=sys.stderr)
        return 1

    y1, m1 = _parse_ym(args.train_month)
    y2, m2 = _parse_ym(args.test_month)

    train = _filter_month(days, y1, m1)
    test = _filter_month(days, y2, m2)
    if not train:
        print(
            f"Error: no trading days in train month {args.train_month} "
            f"(pickle range {days[0]}..{days[-1]})",
            file=sys.stderr,
        )
        return 1
    if not test:
        print(
            f"Error: no trading days in test month {args.test_month} "
            f"(pickle range {days[0]}..{days[-1]})",
            file=sys.stderr,
        )
        return 1

    tr_s, tr_e = train[0], train[-1]
    te_s, te_e = test[0], test[-1]
    if tr_e >= te_s:
        print(
            f"Warning: last train day {tr_e} is on/after first test day {te_s}.",
            file=sys.stderr,
        )

    if args.export_bash:
        print(f"export TRAIN_START_DATE='{tr_s.isoformat()}'")
        print(f"export TRAIN_END_DATE='{tr_e.isoformat()}'")
        print(f"export TEST_START_DATE='{te_s.isoformat()}'")
        print(f"export TEST_END_DATE='{te_e.isoformat()}'")
    else:
        print(f"TRAIN: {tr_s} .. {tr_e} ({len(train)} days)")
        print(f"TEST:  {te_s} .. {te_e} ({len(test)} days)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
