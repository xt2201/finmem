#!/bin/bash
set -e

# Near-paper evaluation split
# Train: 2021-08-17 -> 2022-10-05
# Test : 2022-10-06 -> 2023-04-10

MARKET_MODE="${MARKET_MODE:-${FINMEM_MARKET_MODE:-${FINMEM_MARKET:-US}}}"
MARKET_MODE_UC="$(printf '%s' "$MARKET_MODE" | tr '[:lower:]' '[:upper:]')"
export FINMEM_MARKET_MODE="$MARKET_MODE_UC"
export FINMEM_MARKET="$MARKET_MODE_UC"

if [ -z "${SYMBOL:-}" ]; then
  if [ "$MARKET_MODE_UC" = "VN" ]; then
    SYMBOL="VCI"
  else
    SYMBOL="TSLA"
  fi
fi

SYMBOL_LC="$(printf '%s' "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
export FINMEM_TRADING_SYMBOL="${FINMEM_TRADING_SYMBOL:-$SYMBOL}"

MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/03_model_input/${SYMBOL_LC}.pkl}"
if [ -z "${CONFIG_PATH:-}" ]; then
  if [ "$MARKET_MODE_UC" = "VN" ]; then
    CONFIG_PATH="config/finmem_cerebras_vn_config.toml"
  else
    CONFIG_PATH="config/finmem_cerebras_config.toml"
  fi
fi
TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-data/06_train_checkpoint}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-data/05_train_model_output}"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-data/08_test_checkpoint}"
TEST_OUTPUT="${TEST_OUTPUT:-data/09_results}"
TRAIN_START_DATE="${TRAIN_START_DATE:-2021-08-17}"
TRAIN_END_DATE="${TRAIN_END_DATE:-2022-10-05}"
TEST_START_DATE="${TEST_START_DATE:-2022-10-06}"
TEST_END_DATE="${TEST_END_DATE:-2023-04-10}"
RL_EPISODES="${RL_EPISODES:-20}"
RL_WINDOW="${RL_WINDOW:-10}"
RL_TRANSACTION_COST="${RL_TRANSACTION_COST:-0.001}"
RL_SEED="${RL_SEED:-42}"
RL_RETRY_COUNT="${RL_RETRY_COUNT:-2}"
RL_RETRY_SEED_STEP="${RL_RETRY_SEED_STEP:-101}"
RL_ACTIONS_OUTPUT_DIR="${RL_ACTIONS_OUTPUT_DIR:-$TEST_OUTPUT}"
METRICS_OUTPUT_PATH="${METRICS_OUTPUT_PATH:-$TEST_OUTPUT/${SYMBOL}_metrics_5measures.csv}"
FIGURE_OUTPUT_PATH="${FIGURE_OUTPUT_PATH:-$TEST_OUTPUT/${SYMBOL}_5measures.png}"
PYTHON_BIN="${PYTHON_BIN:-}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "../.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="../.venv/Scripts/python.exe"
  elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if [ ! -f "$MARKET_DATA_PATH" ]; then
  echo "Missing market data file: $MARKET_DATA_PATH"
  echo "Build it first with: .venv/bin/python data-pipeline/09_build_paper_input.py --symbol $SYMBOL"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing config file: $CONFIG_PATH"
  exit 1
fi

echo "Using config: $CONFIG_PATH"

mkdir -p "$TRAIN_CHECKPOINT" "$TRAIN_OUTPUT" "$TEST_CHECKPOINT" "$TEST_OUTPUT" data/04_model_output_log

"$PYTHON_BIN" run.py sim \
  -mdp "$MARKET_DATA_PATH" \
  -st "$TRAIN_START_DATE" \
  -et "$TRAIN_END_DATE" \
  -rm train \
  -mm "$MARKET_MODE_UC" \
  -cp "$CONFIG_PATH" \
  -ckp "$TRAIN_CHECKPOINT" \
  -rp "$TRAIN_OUTPUT" \
  --trading-symbol "$SYMBOL"

"$PYTHON_BIN" run.py sim \
  -mdp "$MARKET_DATA_PATH" \
  -st "$TEST_START_DATE" \
  -et "$TEST_END_DATE" \
  -rm test \
  -mm "$MARKET_MODE_UC" \
  -cp "$CONFIG_PATH" \
  -tap "$TRAIN_OUTPUT" \
  -ckp "$TEST_CHECKPOINT" \
  -rp "$TEST_OUTPUT" \
  --trading-symbol "$SYMBOL"

"$PYTHON_BIN" run.py sim-rl \
  --algorithm all \
  --market-data-path "$MARKET_DATA_PATH" \
  --train-start "$TRAIN_START_DATE" \
  --train-end "$TRAIN_END_DATE" \
  --test-start "$TEST_START_DATE" \
  --test-end "$TEST_END_DATE" \
  --episodes "$RL_EPISODES" \
  --window "$RL_WINDOW" \
  --transaction-cost "$RL_TRANSACTION_COST" \
  --trading-symbol "$SYMBOL" \
  --market-mode "$MARKET_MODE_UC" \
  --seed "$RL_SEED" \
  --retry-count "$RL_RETRY_COUNT" \
  --retry-seed-step "$RL_RETRY_SEED_STEP" \
  --finmem-state-dict "$TEST_OUTPUT/agent_1/state_dict.pkl" \
  --actions-output-dir "$RL_ACTIONS_OUTPUT_DIR" \
  --save-path "$FIGURE_OUTPUT_PATH"

"$PYTHON_BIN" data-pipeline/07-metrics.py \
  --ticker "$SYMBOL" \
  --market "$MARKET_MODE_UC" \
  --start "$TEST_START_DATE" \
  --end "$TEST_END_DATE" \
  --market-data-path "$MARKET_DATA_PATH" \
  --state-dict-path "$TEST_OUTPUT/agent_1/state_dict.pkl" \
  --actions-output-dir "$RL_ACTIONS_OUTPUT_DIR" \
  --save-path "$METRICS_OUTPUT_PATH" \
  --require-five-measures

"$PYTHON_BIN" data-pipeline/06-Visualize-results.py \
  --ticker "$SYMBOL" \
  --market "$MARKET_MODE_UC" \
  --start "$TEST_START_DATE" \
  --end "$TEST_END_DATE" \
  --market-data-path "$MARKET_DATA_PATH" \
  --state-dict-path "$TEST_OUTPUT/agent_1/state_dict.pkl" \
  --actions-output-dir "$RL_ACTIONS_OUTPUT_DIR" \
  --save-path "$FIGURE_OUTPUT_PATH" \
  --require-five-measures

echo "Near-paper 5-measure evaluation completed for symbol: $SYMBOL (market: $MARKET_MODE_UC)"
echo "Metrics: $METRICS_OUTPUT_PATH"
echo "Figure:  $FIGURE_OUTPUT_PATH"
