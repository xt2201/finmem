"""Test 03: Memory Functions (scoring, decay, access counter, recency)
Validates:
- Importance score initialization distributions per layer
- Recency initialization (always 1.0)
- Exponential decay behavior
- LinearCompoundScore calculation
- LinearImportanceScoreChange (access counter feedback)
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
results = {"test": "test_03_memory_functions", "passed": [], "failed": []}


def run():
    from puppy.memory_functions import (
        get_importance_score_initialization_func,
        R_ConstantInitialization,
        LinearCompoundScore,
        ExponentialDecay,
        LinearImportanceScoreChange,
    )

    # 1. Importance score initialization per layer
    for layer in ["short", "mid", "long"]:
        func = get_importance_score_initialization_func(type="sample", memory_layer=layer)
        scores = [func() for _ in range(200)]
        unique = set(scores)
        results[f"importance_init_{layer}"] = {"unique_values": sorted(unique), "sample_size": len(scores)}
        if len(unique) >= 2:
            results["passed"].append(f"importance_init_{layer}_distribution")
        else:
            results["failed"].append({f"importance_init_{layer}": f"only {unique}"})

    # 2. Recency initialization
    recency = R_ConstantInitialization()
    val = recency()
    if val == 1.0:
        results["passed"].append("recency_init_1.0")
    else:
        results["failed"].append({"recency_init": f"expected 1.0, got {val}"})

    # 3. Exponential decay — signature: (important_score, delta) -> (recency, importance, delta)
    decay = ExponentialDecay(recency_factor=3.0, importance_factor=0.92)
    r, i, d = decay(important_score=50.0, delta=0)
    results["decay_step_0"] = {"recency": float(r), "importance": float(i), "delta": float(d)}
    if d == 1 and i < 50.0 and 0 < r < 1.0:
        results["passed"].append("decay_step_0")
    else:
        results["failed"].append({"decay_step_0": f"r={r}, i={i}, d={d}"})

    # Multi-step decay
    cur_i, cur_d = i, d
    for step in range(1, 5):
        r, cur_i, cur_d = decay(important_score=cur_i, delta=cur_d)
    results["decay_step_5"] = {"recency": float(r), "importance": float(cur_i), "delta": float(cur_d)}
    expected_max = 50.0 * (0.92 ** 5)
    if cur_d == 5 and cur_i <= expected_max:
        results["passed"].append("decay_multi_step")
    else:
        results["failed"].append({"decay_multi_step": f"i={cur_i}, d={cur_d}, expected i < {expected_max}"})

    # 4. LinearCompoundScore — merge_score(similarity_score, recency_and_importance)
    compound = LinearCompoundScore()
    partial = compound.recency_and_importance_score(recency_score=0.8, importance_score=60.0)
    results["compound_partial"] = partial
    expected_partial = 0.8 + 60.0 / 100
    if abs(partial - expected_partial) < 1e-9:
        results["passed"].append("compound_partial_score")
    else:
        results["failed"].append({"compound_partial": f"expected {expected_partial}, got {partial}"})

    merged = compound.merge_score(similarity_score=0.9, recency_and_importance=1.4)
    results["compound_merged"] = merged
    if abs(merged - 2.3) < 1e-9:
        results["passed"].append("compound_merge_score")
    else:
        results["failed"].append({"compound_merge": f"expected 2.3, got {merged}"})

    # 5. LinearImportanceScoreChange
    change = LinearImportanceScoreChange()
    new_score = change(access_counter=2, importance_score=50.0)
    results["importance_change"] = {"input": 50.0, "access_counter": 2, "output": new_score}
    if abs(new_score - 60.0) < 1e-9:
        results["passed"].append("importance_score_change")
    else:
        results["failed"].append({"importance_change": f"expected 60.0, got {new_score}"})

    save_and_exit()


def save_and_exit():
    results["summary"] = f"{len(results['passed'])} passed, {len(results['failed'])} failed"
    with open(OUTPUT_DIR / "test_03_memory_functions.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
