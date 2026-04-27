#!/usr/bin/env bash
# Load several env files in order; later values override. Handles KEY = "value" style lines
# via the same parser as the Python tooling (python-dotenv).
#
# Usage:
#   set -a
#   source scripts/source_env_stack.sh .env .env.linh
#   set +a
set -e
_THIS="${BASH_SOURCE[0]:-$0}"
ROOT="$(cd "$(dirname "$_THIS")/.." && pwd)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi
eval "$(
  "$PY" -c "
import os, sys, shlex
from dotenv import dotenv_values
merged: dict = {}
for path in sys.argv[1:]:
    if not path or not os.path.isfile(path):
        continue
    merged.update({k: v for k, v in dotenv_values(path).items() if v is not None and k})
for k, v in merged.items():
    print('export ' + k + '=' + shlex.quote(v))
" "$@"
)"
