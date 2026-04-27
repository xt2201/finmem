#!/usr/bin/env bash
# Run `run_paper_eval.sh` with per-symbol checkpoint/output paths so US/VN sweeps
# do not clobber the default `data/06_train_checkpoint` / `data/09_results` trees.
#
# US (paper dates from README; uses defaults in run_paper_eval.sh):
#   SYMBOL=MSFT MARKET_MODE=US bash scripts/run_isolated_paper_eval.sh
#
# VN: set VN_TRAIN_MONTH / VN_TEST_MONTH (YYYY-MM) to auto-fill date envs from the pickle.
#   SYMBOL=BID MARKET_MODE=VN VN_TRAIN_MONTH=2025-04 VN_TEST_MONTH=2025-05 bash scripts/run_isolated_paper_eval.sh
# Build VN data first with FINMEM_VN_TRANSLATE_FOR_VADER=1 (see README).
#
# If a provider returns quota errors, switch keys in the environment and re-run, e.g.:
#   export CEREBRAS_API_KEY="..."
#   export ALPACA_API_KEY="..."  # for US builds; sim uses LLM + RL as configured

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MARKET_MODE="${MARKET_MODE:-${FINMEM_MARKET_MODE:-${FINMEM_MARKET:-US}}}"
export MARKET_MODE
MARKET_MODE_UC="$(printf '%s' "$MARKET_MODE" | tr '[:lower:]' '[:upper:]')"
export FINMEM_MARKET_MODE="$MARKET_MODE_UC"
export FINMEM_MARKET="$MARKET_MODE_UC"

if [ -z "${SYMBOL:-}" ]; then
  echo "Set SYMBOL, e.g. SYMBOL=NFLX" >&2
  exit 1
fi

SYMBOL_LC="$(printf '%s' "$SYMBOL" | tr '[:upper:]' '[:lower:]')"
SUF="_${SYMBOL_LC}"

if [ -z "${CONFIG_PATH:-}" ]; then
  if [ "$MARKET_MODE_UC" = "VN" ]; then
    CONFIG_PATH="config/finmem_openrouter_vn_config.toml"
  else
    CONFIG_PATH="config/finmem_openrouter_config.toml"
  fi
fi
export CONFIG_PATH

export MARKET_DATA_PATH="${MARKET_DATA_PATH:-data/03_model_input/${SYMBOL_LC}.pkl}"
export TRAIN_CHECKPOINT="${TRAIN_CHECKPOINT:-data/06_train_checkpoint${SUF}}"
export TRAIN_OUTPUT="${TRAIN_OUTPUT:-data/05_train_model_output${SUF}}"
export TEST_CHECKPOINT="${TEST_CHECKPOINT:-data/08_test_checkpoint${SUF}}"
export TEST_OUTPUT="${TEST_OUTPUT:-data/09_results${SUF}}"
export RL_ACTIONS_OUTPUT_DIR="${RL_ACTIONS_OUTPUT_DIR:-$TEST_OUTPUT}"
export METRICS_OUTPUT_PATH="${METRICS_OUTPUT_PATH:-$TEST_OUTPUT/${SYMBOL}_metrics_5measures.csv}"
export FIGURE_OUTPUT_PATH="${FIGURE_OUTPUT_PATH:-$TEST_OUTPUT/${SYMBOL}_5measures.png}"

if [ "$MARKET_MODE_UC" = "VN" ] && [ -n "${VN_TRAIN_MONTH:-}" ] && [ -n "${VN_TEST_MONTH:-}" ]; then
  eval "$(.venv/bin/python scripts/vn_train_test_from_pkl.py \
    "$MARKET_DATA_PATH" --train-month "$VN_TRAIN_MONTH" --test-month "$VN_TEST_MONTH" --export-bash)"
  export TRAIN_START_DATE TRAIN_END_DATE TEST_START_DATE TEST_END_DATE
fi

exec bash run_paper_eval.sh
