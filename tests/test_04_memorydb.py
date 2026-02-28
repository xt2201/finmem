"""Test 04: MemoryDB
Validates:
- MemoryDB: add_memory, query, decay step, access counter update, checkpoint
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUTPUT_DIR / "memorydb_checkpoint"
results = {"test": "test_04_memorydb", "passed": [], "failed": []}


def run():
    from puppy.memorydb import MemoryDB, id_generator_func
    from puppy.memory_functions import (
        get_importance_score_initialization_func,
        R_ConstantInitialization,
        LinearCompoundScore,
        ExponentialDecay,
        LinearImportanceScoreChange,
    )

    logger = logging.getLogger("test_memorydb")
    logger.addHandler(logging.NullHandler())
    id_gen = id_generator_func()

    emb_config = {
        "embedding_model": "intfloat/multilingual-e5-large",
        "chunk_size": 5000,
        "verbose": False,
    }

    # 1. MemoryDB creation
    try:
        mdb = MemoryDB(
            db_name="test_short",
            id_generator=id_gen,
            jump_threshold_upper=60,
            jump_threshold_lower=-999999,
            logger=logger,
            emb_config=emb_config,
            importance_score_initialization=get_importance_score_initialization_func("sample", "short"),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(recency_factor=3.0, importance_factor=0.92),
            clean_up_threshold_dict={"recency_threshold": 0.05, "importance_threshold": 5},
        )
        results["passed"].append("memorydb_create")
    except Exception as e:
        results["failed"].append({"memorydb_create": str(e)})
        save_and_exit()
        return

    # 2. Add single memory
    try:
        mdb.add_memory("TSLA", date(2022, 7, 14), ["Tesla earnings beat expectations this quarter"])
        mem_count = len(mdb.universe["TSLA"]["score_memory"])
        results["add_memory_count"] = mem_count
        if mem_count == 1:
            results["passed"].append("add_single_memory")
        else:
            results["failed"].append({"add_single_memory": f"count={mem_count}"})
    except Exception as e:
        results["failed"].append({"add_memory": str(e)})
        save_and_exit()
        return

    # 3. Add batch of memories
    mdb.add_memory("TSLA", date(2022, 7, 15), [
        "Tesla production numbers increased",
        "Tesla autopilot update released",
    ])
    mem_count = len(mdb.universe["TSLA"]["score_memory"])
    if mem_count == 3:
        results["passed"].append("add_batch_memory")
    else:
        results["failed"].append({"add_batch_memory": f"expected 3, got {mem_count}"})

    # 4. Query memories
    try:
        texts, ids = mdb.query("Tesla financial performance", top_k=2, symbol="TSLA")
        results["query_results"] = {"count": len(texts), "ids": ids}
        if len(texts) >= 1 and len(ids) >= 1:
            results["passed"].append("query_memory")
        else:
            results["failed"].append({"query_memory": "empty results"})
    except Exception as e:
        results["failed"].append({"query_memory": str(e)})
        ids = []

    # 5. Decay step
    try:
        removed = mdb.step()
        results["decay_removed_ids"] = list(removed) if removed else []
        results["passed"].append("decay_step")
    except Exception as e:
        results["failed"].append({"decay_step": str(e)})

    # 6. Access counter update
    if ids:
        try:
            success_ids = mdb.update_access_count_with_feed_back("TSLA", ids[:1], [1])
            if len(success_ids) >= 1:
                results["passed"].append("access_counter_update")
            else:
                results["failed"].append({"access_counter": "no ids updated"})
        except Exception as e:
            results["failed"].append({"access_counter": str(e)})

    # 7. Checkpoint save/load round-trip
    try:
        if CKPT_DIR.exists():
            shutil.rmtree(CKPT_DIR)
        CKPT_DIR.mkdir(parents=True)
        pre_count = len(mdb.universe["TSLA"]["score_memory"])
        mdb.save_checkpoint("test_layer", str(CKPT_DIR), force=True)
        loaded = MemoryDB.load_checkpoint(str(CKPT_DIR / "test_layer"))
        loaded_count = len(loaded.universe["TSLA"]["score_memory"])
        if loaded_count == pre_count:
            results["passed"].append("checkpoint_roundtrip")
        else:
            results["failed"].append({"checkpoint": f"saved={pre_count}, loaded={loaded_count}"})
    except Exception as e:
        results["failed"].append({"checkpoint": str(e)})

    save_and_exit()


def save_and_exit():
    results["summary"] = f"{len(results['passed'])} passed, {len(results['failed'])} failed"
    with open(OUTPUT_DIR / "test_04_memorydb.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
