#!/bin/bash
set -e

export CEREBRAS_API_KEY="${CEREBRAS_API_KEY:-Enter your Cerebras API Key here}"

SYMBOL="${SYMBOL:-TSLA}"
SYMBOL_LC="$(printf '%s' "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
RUN_MODE="${RUN_MODE:-test}"

export FINMEM_TRADING_SYMBOL="${FINMEM_TRADING_SYMBOL:-$SYMBOL}"
MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/03_model_input/${SYMBOL_LC}.pkl}"
CONFIG_PATH="${CONFIG_PATH:-config/finmem_cerebras_config.toml}"
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
    -cp "$CONFIG_PATH" \
    -ckp "$CHECKPOINT_PATH" \
    -rp "$RESULT_PATH"
else
  python run.py sim \
    -mdp "$MARKET_DATA_PATH" \
    -st "$START_DATE" \
    -et "$END_DATE" \
    -rm test \
    -cp "$CONFIG_PATH" \
    -tap "$TRAINED_AGENT_PATH" \
    -ckp "$CHECKPOINT_PATH" \
    -rp "$RESULT_PATH"
fi

python save_file.py --agent-checkpoint "${SAVE_AGENT_CHECKPOINT:-$CHECKPOINT_PATH/agent_1}" --ticker "$SYMBOL"
