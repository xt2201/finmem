"""Runner: execute all test scripts sequentially and produce a summary."""
import subprocess
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TESTS_DIR / "outputs"
VENV_PYTHON = str(TESTS_DIR.parent / ".venv" / "bin" / "python")

test_files = sorted(TESTS_DIR.glob("test_*.py"))

overall = {"total": 0, "passed": 0, "failed": 0, "errors": [], "details": {}}

for tf in test_files:
    name = tf.stem
    overall["total"] += 1
    result = subprocess.run(
        [VENV_PYTHON, str(tf)],
        cwd=str(TESTS_DIR.parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    output_file = OUTPUT_DIR / f"{name}.json"
    if result.returncode != 0:
        overall["errors"].append({"test": name, "stderr": result.stderr[-500:]})
        overall["details"][name] = "ERROR"
    elif output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        p = len(data.get("passed", []))
        fl = len(data.get("failed", []))
        overall["passed"] += p
        overall["failed"] += fl
        overall["details"][name] = data.get("summary", f"{p}p/{fl}f")
    else:
        overall["errors"].append({"test": name, "note": "no output file"})

overall["grand_summary"] = f"{overall['passed']} passed, {overall['failed']} failed, {len(overall['errors'])} errors"

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(overall, f, indent=2)

sys.stdout.write(json.dumps(overall, indent=2) + "\n")
