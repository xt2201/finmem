#!/bin/bash
set -e

# Near-paper evaluation split for TSLA
# Train: 2021-08-17 -> 2022-10-05
# Test : 2022-10-06 -> 2023-04-10

MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/03_model_input/tsla.pkl}"
CONFIG_PATH="${CONFIG_PATH:-config/tsla_cerebras_config.toml}"
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-data/06_train_checkpoint}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-data/05_train_model_output}"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-data/08_test_checkpoint}"
TEST_OUTPUT="${TEST_OUTPUT:-data/09_results}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if [ ! -f "$MARKET_DATA_PATH" ]; then
  echo "Missing market data file: $MARKET_DATA_PATH"
  echo "Build it first with: .venv/bin/python data-pipeline/09_build_paper_tsla_input.py"
  exit 1
fi

"$PYTHON_BIN" run.py sim \
  -mdp "$MARKET_DATA_PATH" \
  -st 2021-08-17 \
  -et 2022-10-05 \
  -rm train \
  -cp "$CONFIG_PATH" \
  -ckp "$TRAIN_CHECKPOINT" \
  -rp "$TRAIN_OUTPUT"

"$PYTHON_BIN" run.py sim \
  -mdp "$MARKET_DATA_PATH" \
  -st 2022-10-06 \
  -et 2023-04-10 \
  -rm test \
  -cp "$CONFIG_PATH" \
  -tap "$TRAIN_OUTPUT" \
  -ckp "$TEST_CHECKPOINT" \
  -rp "$TEST_OUTPUT"

echo "Near-paper TSLA evaluation completed."
