#!/usr/bin/env bash
# Chạy đủ pipeline README cho US: NFLX, AMZN, MSFT và VN: BID, MBB, FPT
# (build dữ liệu nếu thiếu → run_paper_eval: train + test + RL + metrics + plot).
#
# Build và eval cho từng mã chạy SONG SONG (6 tiến trình / phase). Cần đủ RAM/VRAM;
# giảm tải: chỉnh script hoặc tạm export CUDA_VISIBLE_DEVICES trống cho một phần job.
#
# Tùy chọn môi trường:
#   FORCE_REBUILD=1  — luôn chạy lại 09_build_paper_input.py (tốn API).
#   SKIP_BUILD=1     — bỏ qua bước build, chỉ chạy eval (cần pickle sẵn).
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

US_START="2021-08-17"
US_END="2023-04-10"

wait_pids_or_fail() {
  local ec=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      ec=1
    fi
  done
  return "$ec"
}

build_us() {
  local sym="$1"
  local out="data/03_model_input/$(printf '%s' "$sym" | tr '[:upper:]' '[:lower:]').pkl"
  if [ -n "${SKIP_BUILD:-}" ] && [ "${SKIP_BUILD}" = "1" ]; then
    echo "[SKIP_BUILD] bỏ qua US $sym"
    return 0
  fi
  if [ -f "$out" ] && [ -z "${FORCE_REBUILD:-}" ]; then
    echo "[OK] đã có $out — bỏ qua build (FORCE_REBUILD=1 để build lại)"
    return 0
  fi
  echo "==== README: Build market input (US) $sym ===="
  FINMEM_MARKET_MODE=US FINMEM_MARKET=US "$PY" data-pipeline/09_build_paper_input.py \
    --market US --symbol "$sym" --start "$US_START" --end "$US_END"
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
  echo "==== README: Build market input (VN) $sym $st .. $en (VADER + dịch nếu bật env) ===="
  export FINMEM_MARKET_MODE=VN
  export FINMEM_MARKET=VN
  export FINMEM_VNSTOCK_SOURCE=KBS
  export FINMEM_VN_TRANSLATE_FOR_VADER="${FINMEM_VN_TRANSLATE_FOR_VADER:-1}"
  "$PY" data-pipeline/09_build_paper_input.py \
    --market VN --symbol "$sym" --start "$st" --end "$en"
}

eval_us() {
  local sym="$1"
  echo "==== README: Near-paper eval (US) $sym ===="
  export FINMEM_VN_TRANSLATE_FOR_VADER="${FINMEM_VN_TRANSLATE_FOR_VADER:-}"
  SYMBOL="$sym" MARKET_MODE=US bash "${ROOT}/scripts/run_isolated_paper_eval.sh"
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

echo "===== Bước 1 (README): Build market input data (song song) ====="
bpids=()
build_us NFLX & bpids+=($!)
build_us AMZN & bpids+=($!)
build_us MSFT & bpids+=($!)
# BID 04-05/2025; MBB 07-08/2025; FPT 05-06/2025
build_vn BID "2025-04-01" "2025-05-31" & bpids+=($!)
build_vn MBB "2025-07-01" "2025-08-31" & bpids+=($!)
build_vn FPT "2025-05-01" "2025-06-30" & bpids+=($!)
if ! wait_pids_or_fail "${bpids[@]}"; then
  echo "Lỗi: ít nhất một job build thất bại." >&2
  exit 1
fi

echo "===== Bước 2–4 (README): Train, test, RL, metrics, visualize (song song) ====="
epids=()
eval_us NFLX & epids+=($!)
eval_us AMZN & epids+=($!)
eval_us MSFT & epids+=($!)
eval_vn BID "2025-04" "2025-05" & epids+=($!)
eval_vn MBB "2025-07" "2025-08" & epids+=($!)
eval_vn FPT "2025-05" "2025-06" & epids+=($!)
if ! wait_pids_or_fail "${epids[@]}"; then
  echo "Lỗi: ít nhất một job eval thất bại." >&2
  exit 1
fi

echo "===== Hoàn tất 6 mã. Kết quả: data/09_results_<ticker>/ ====="
