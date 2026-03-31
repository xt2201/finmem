#!/bin/bash
set -e

export OPENAI_API_KEY="${OPENAI_API_KEY:-Enter your OpenAI API Key here}"

SYMBOL="${SYMBOL:-TSLA}"
SYMBOL_LC="$(printf '%s' "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
RUN_MODE="${RUN_MODE:-train}"

export FINMEM_TRADING_SYMBOL="${FINMEM_TRADING_SYMBOL:-$SYMBOL}"
MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/03_model_input/${SYMBOL_LC}.pkl}"
CONFIG_PATH="${CONFIG_PATH:-config/finmem_gemini_config.toml}"
TRAINED_AGENT_PATH="${TRAINED_AGENT_PATH:-./data/06_train_checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./data/06_train_checkpoint}"
RESULT_PATH="${RESULT_PATH:-./data/05_train_model_output}"
START_DATE="${START_DATE:-2022-07-21}"
END_DATE="${END_DATE:-2022-10-07}"

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
