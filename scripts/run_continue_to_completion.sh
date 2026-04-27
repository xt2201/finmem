#!/usr/bin/env bash
# Tiếp tục từ checkpoint NFLX (nếu có), chạy tới hết: NFLX (test+RL+metrics) → AMZN → MSFT → BID → MBB → FPT
set -euo pipefail

THIS="${BASH_SOURCE[0]:-$0}"
ROOT="$(cd "$(dirname "$THIS")/.." && pwd)"
cd "$ROOT"

if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT}/scripts/source_env_stack.sh" "${ROOT}/.env"
  set +a
fi

if [ -x "${ROOT}/.venv/bin/python" ]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="python3"
fi

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

run_paper_tail () {
  # Từ run_paper_eval: sim test, sim-rl, 07, 06 (với env đã set trước: SYMBOL, paths, dates)
  "$PY" run.py sim \
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

  "$PY" run.py sim-rl \
    --algorithm all \
    --market-data-path "$MARKET_DATA_PATH" \
    --train-start "$TRAIN_START_DATE" \
    --train-end "$TRAIN_END_DATE" \
    --test-start "$TEST_START_DATE" \
    --test-end "$TEST_END_DATE" \
    --episodes "${RL_EPISODES:-20}" \
    --window "${RL_WINDOW:-10}" \
    --transaction-cost "${RL_TRANSACTION_COST:-0.001}" \
    --trading-symbol "$SYMBOL" \
    --market-mode "$MARKET_MODE_UC" \
    --seed "${RL_SEED:-42}" \
    --retry-count "${RL_RETRY_COUNT:-2}" \
    --retry-seed-step "${RL_RETRY_SEED_STEP:-101}" \
    --finmem-state-dict "$TEST_OUTPUT/agent_1/state_dict.pkl" \
    --actions-output-dir "$RL_ACTIONS_OUTPUT_DIR" \
    --save-path "$FIGURE_OUTPUT_PATH"

  "$PY" data-pipeline/07-metrics.py \
    --ticker "$SYMBOL" \
    --market "$MARKET_MODE_UC" \
    --start "$TEST_START_DATE" \
    --end "$TEST_END_DATE" \
    --market-data-path "$MARKET_DATA_PATH" \
    --state-dict-path "$TEST_OUTPUT/agent_1/state_dict.pkl" \
    --actions-output-dir "$RL_ACTIONS_OUTPUT_DIR" \
    --save-path "$METRICS_OUTPUT_PATH" \
    --require-five-measures

  "$PY" data-pipeline/06-Visualize-results.py \
    --ticker "$SYMBOL" \
    --market "$MARKET_MODE_UC" \
    --start "$TEST_START_DATE" \
    --end "$TEST_END_DATE" \
    --market-data-path "$MARKET_DATA_PATH" \
    --state-dict-path "$TEST_OUTPUT/agent_1/state_dict.pkl" \
    --actions-output-dir "$RL_ACTIONS_OUTPUT_DIR" \
    --save-path "$FIGURE_OUTPUT_PATH" \
    --require-five-measures
}

# --- 1) NFLX: nối train từ checkpoint, rồi phần còn lại
SYMBOL="NFLX"
MARKET_MODE="US"
MARKET_MODE_UC="US"
export FINMEM_MARKET_MODE=US
export FINMEM_MARKET=US
CONFIG_PATH="config/finmem_cerebras_config.toml"
MARKET_DATA_PATH="data/03_model_input/nflx.pkl"
TRAIN_CHECKPOINT="data/06_train_checkpoint_nflx"
TRAIN_OUTPUT="data/05_train_model_output_nflx"
TEST_CHECKPOINT="data/08_test_checkpoint_nflx"
TEST_OUTPUT="data/09_results_nflx"
RL_ACTIONS_OUTPUT_DIR="${RL_ACTIONS_OUTPUT_DIR:-$TEST_OUTPUT}"
METRICS_OUTPUT_PATH="$TEST_OUTPUT/${SYMBOL}_metrics_5measures.csv"
FIGURE_OUTPUT_PATH="$TEST_OUTPUT/${SYMBOL}_5measures.png"
TRAIN_START_DATE="2021-08-17"
TRAIN_END_DATE="2022-10-05"
TEST_START_DATE="2022-10-06"
TEST_END_DATE="2023-04-10"

mkdir -p "$TRAIN_CHECKPOINT" "$TRAIN_OUTPUT" "$TEST_CHECKPOINT" "$TEST_OUTPUT" data/04_model_output_log

if [ -d "$TRAIN_CHECKPOINT/env" ] && [ -d "$TRAIN_CHECKPOINT/agent_1" ]; then
  echo "==== Nối train NFLX từ sim-checkpoint ===="
  "$PY" run.py sim-checkpoint \
    -ckp "$TRAIN_CHECKPOINT" \
    -rp "$TRAIN_OUTPUT" \
    -cp "$CONFIG_PATH" \
    -rm train \
    -mm US \
    --trading-symbol NFLX
else
  echo "==== Không có checkpoint NFLX, train đầy đủ ===="
  "$PY" run.py sim \
    -mdp "$MARKET_DATA_PATH" \
    -st "$TRAIN_START_DATE" \
    -et "$TRAIN_END_DATE" \
    -rm train \
    -mm US \
    -cp "$CONFIG_PATH" \
    -ckp "$TRAIN_CHECKPOINT" \
    -rp "$TRAIN_OUTPUT" \
    --trading-symbol NFLX
fi

echo "==== NFLX: test + RL + metrics + figure ===="
run_paper_tail

echo "==== NFLX xong. Chạy AMZN, MSFT, VN ===="

# --- 2) AMZN, MSFT (full paper eval isolated)
for S in AMZN MSFT; do
  SYMBOL="$S" MARKET_MODE=US bash "${ROOT}/scripts/run_isolated_paper_eval.sh"
done

# --- 3) VN
export FINMEM_VN_TRANSLATE_FOR_VADER="${FINMEM_VN_TRANSLATE_FOR_VADER:-1}"
export FINMEM_VNSTOCK_SOURCE=KBS

SYMBOL="BID" MARKET_MODE=VN VN_TRAIN_MONTH=2025-04 VN_TEST_MONTH=2025-05 bash "${ROOT}/scripts/run_isolated_paper_eval.sh"
SYMBOL="MBB" MARKET_MODE=VN VN_TRAIN_MONTH=2025-07 VN_TEST_MONTH=2025-08 bash "${ROOT}/scripts/run_isolated_paper_eval.sh"
SYMBOL="FPT" MARKET_MODE=VN VN_TRAIN_MONTH=2025-05 VN_TEST_MONTH=2025-06 bash "${ROOT}/scripts/run_isolated_paper_eval.sh"

echo "==== Hoàn tất 6 mã. Artifacts: data/09_results_*/  ===="
