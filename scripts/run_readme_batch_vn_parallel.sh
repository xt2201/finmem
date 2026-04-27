#!/usr/bin/env bash
# Chỉ VN: BID, MBB, FPT — build song song → eval song song (README date ranges).
#
#   FORCE_REBUILD=1  — luôn build lại pickle
#   SKIP_BUILD=1     — chỉ eval (cần data/03_model_input/<sym>.pkl)
#
set -euo pipefail

THIS="${BASH_SOURCE[0]:-$0}"
ROOT="$(cd "$(dirname "$THIS")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${ROOT}/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT}/scripts/source_env_stack.sh" "${ROOT}/.env"
  set +a
fi

wait_pids_or_fail() {
  local ec=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      ec=1
    fi
  done
  return "$ec"
}

build_vn() {
  local sym="$1" st="$2" en="$3"
  local out="data/03_model_input/$(printf '%s' "$sym" | tr '[:upper:]' '[:lower:]').pkl"
  if [ -n "${SKIP_BUILD:-}" ] && [ "${SKIP_BUILD}" = "1" ]; then
    echo "[SKIP_BUILD] bỏ qua VN $sym"
    return 0
  fi
  if [ -f "$out" ] && [ -z "${FORCE_REBUILD:-}" ]; then
    echo "[OK] đã có $out — bỏ qua build (FORCE_REBUILD=1 để build lại)"
    return 0
  fi
  echo "==== README: Build (VN) $sym $st .. $en ===="
  export FINMEM_MARKET_MODE=VN
  export FINMEM_MARKET=VN
  export FINMEM_VNSTOCK_SOURCE=KBS
  export FINMEM_VN_TRANSLATE_FOR_VADER="${FINMEM_VN_TRANSLATE_FOR_VADER:-1}"
  "$PY" data-pipeline/09_build_paper_input.py \
    --market VN --symbol "$sym" --start "$st" --end "$en"
}

eval_vn() {
  local sym="$1" tm="$2" tst="$3"
  echo "==== README: Near-paper eval (VN) $sym (train $tm / test $tst) ===="
  export FINMEM_VN_TRANSLATE_FOR_VADER="${FINMEM_VN_TRANSLATE_FOR_VADER:-1}"
  export FINMEM_MARKET_MODE=VN
  export FINMEM_MARKET=VN
  export FINMEM_VNSTOCK_SOURCE=KBS
  SYMBOL="$sym" MARKET_MODE=VN VN_TRAIN_MONTH="$tm" VN_TEST_MONTH="$tst" \
    bash "${ROOT}/scripts/run_isolated_paper_eval.sh"
}

echo "===== VN only: Build song song (BID, MBB, FPT) ====="
bpids=()
build_vn BID "2025-04-01" "2025-05-31" & bpids+=($!)
build_vn MBB "2025-07-01" "2025-08-31" & bpids+=($!)
build_vn FPT "2025-05-01" "2025-06-30" & bpids+=($!)
if ! wait_pids_or_fail "${bpids[@]}"; then
  echo "Lỗi: ít nhất một job build VN thất bại." >&2
  exit 1
fi

echo "===== VN only: Eval song song ====="
epids=()
eval_vn BID "2025-04" "2025-05" & epids+=($!)
eval_vn MBB "2025-07" "2025-08" & epids+=($!)
eval_vn FPT "2025-05" "2025-06" & epids+=($!)
if ! wait_pids_or_fail "${epids[@]}"; then
  echo "Lỗi: ít nhất một job eval VN thất bại." >&2
  exit 1
fi

echo "===== Hoàn tất 3 mã VN. Kết quả: data/09_results_bid|mbb|fpt/ ====="
