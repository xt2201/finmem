#!/bin/bash
set -e

export CEREBRAS_API_KEY="${CEREBRAS_API_KEY:-Enter your Cerebras API Key here}"

RUN_MODE="${RUN_MODE:-test}"
MARKET_MODE="${MARKET_MODE:-${FINMEM_MARKET_MODE:-${FINMEM_MARKET:-US}}}"
MARKET_MODE_UC="$(printf '%s' "$MARKET_MODE" | tr '[:lower:]' '[:upper:]')"

if [ -z "${SYMBOL:-}" ]; then
  if [ "$MARKET_MODE_UC" = "VN" ]; then
    SYMBOL="VCI"
  else
    SYMBOL="TSLA"
  fi
fi

SYMBOL_LC="$(printf '%s' "$SYMBOL" | tr '[:upper:]' '[:lower:]')"

export FINMEM_TRADING_SYMBOL="${FINMEM_TRADING_SYMBOL:-$SYMBOL}"
export FINMEM_MARKET_MODE="$MARKET_MODE_UC"
export FINMEM_MARKET="$MARKET_MODE_UC"
MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/03_model_input/${SYMBOL_LC}.pkl}"
if [ -z "${CONFIG_PATH:-}" ]; then
  if [ "$MARKET_MODE_UC" = "VN" ]; then
    CONFIG_PATH="config/finmem_cerebras_vn_config.toml"
  else
    CONFIG_PATH="config/finmem_cerebras_config.toml"
  fi
fi
TRAINED_AGENT_PATH="${TRAINED_AGENT_PATH:-./data/06_train_checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./data/08_test_checkpoint}"
RESULT_PATH="${RESULT_PATH:-./data/09_results}"
START_DATE="${START_DATE:-2022-07-20}"
END_DATE="${END_DATE:-2022-08-01}"

if [ "$RUN_MODE" = "train" ]; then
  python run.py sim \
    -mdp "$MARKET_DATA_PATH" \
    -st "$START_DATE" \
    -et "$END_DATE" \
    -rm train \
    -mm "$MARKET_MODE_UC" \
    -cp "$CONFIG_PATH" \
    -ckp "$CHECKPOINT_PATH" \
    -rp "$RESULT_PATH"
else
  python run.py sim \
    -mdp "$MARKET_DATA_PATH" \
    -st "$START_DATE" \
    -et "$END_DATE" \
    -rm test \
    -mm "$MARKET_MODE_UC" \
    -cp "$CONFIG_PATH" \
    -tap "$TRAINED_AGENT_PATH" \
    -ckp "$CHECKPOINT_PATH" \
    -rp "$RESULT_PATH"
fi

python save_file.py --agent-checkpoint "${SAVE_AGENT_CHECKPOINT:-$CHECKPOINT_PATH/agent_1}" --ticker "$SYMBOL"
