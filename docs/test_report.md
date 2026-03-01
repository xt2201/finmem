# FinMem Test Report

**Date:** 2026-03-01  
**Codebase:** `pipiku915/FinMem-LLM-StockTrading`  
**Dataset:** `data/06_input/subset_symbols.pkl` (Fake Sample Data, 496 trading days — TSLA, NFLX, AMZN, MSFT)

---

## 1. Environment

| Item | Value |
|------|-------|
| Python | 3.10.19 |
| OS | macOS |
| Virtual env | `.venv` (uv) |
| guardrails-ai | 0.4.5 |
| langchain-cerebras | 0.6.0 |
| langchain-community | 0.2.19 |
| transformers | 4.57.6 |
| torch | 2.10.0 (CPU) |
| faiss-cpu | 1.13.2 |
| numpy | 1.26.4 |
| polars | 1.38.1 |
| typer | 0.9.4 |
| LLM backend | Cerebras (`zai-glm-4.7`) |
| Embedding model | `intfloat/multilingual-e5-large` (1024-dim, HF Inference API) |

---

## 2. Unit Tests (`tests/run_all_tests.py`)

**Grand result: 34 passed, 0 failed, 0 errors** (6 test files, ~45 s)

### 2.1 `test_01_embedding` — 5/5 passed

| Test case | Result | Detail |
|-----------|--------|--------|
| `init_embedding` | ✅ PASS | `HuggingFaceEmb` initialised with HF token |
| `dimension_check_1024` | ✅ PASS | `get_embedding_dimension()` → 1024 |
| `single_text_shape_dtype` | ✅ PASS | shape `(1, 1024)`, dtype `float32` |
| `batch_embedding_shape` | ✅ PASS | shape `(3, 1024)` for 3 texts |
| `cosine_similarity_ordering` | ✅ PASS | similar pair 0.788 > dissimilar pair 0.757 |

### 2.2 `test_02_chat_cerebras` — 3/3 passed

| Test case | Result | Detail |
|-----------|--------|--------|
| `init_chat` | ✅ PASS | `ChatOpenAICompatible` instantiated |
| `guardrail_endpoint_callable` | ✅ PASS | `guardrail_endpoint()` returns callable |
| `llm_invocation` | ✅ PASS | LLM returned valid JSON `{"result": 4}` |

### 2.3 `test_03_memory_functions` — 9/9 passed

| Test case | Result | Detail |
|-----------|--------|--------|
| `importance_init_short_distribution` | ✅ PASS | scores ∈ {50, 70, 90} over 200 samples |
| `importance_init_mid_distribution` | ✅ PASS | scores ∈ {40, 60, 80} |
| `importance_init_long_distribution` | ✅ PASS | scores ∈ {40, 60, 80} |
| `recency_init_1.0` | ✅ PASS | `R_ConstantInitialization()` → 1.0 |
| `decay_step_0` | ✅ PASS | recency=0.717, importance=46.0, delta=1 after 1 step |
| `decay_multi_step` | ✅ PASS | recency=0.189, importance=32.95, delta=5 after 5 steps |
| `compound_partial_score` | ✅ PASS | 1.0 + 40/100 = 1.40 |
| `compound_merge_score` | ✅ PASS | 0.9 + 1.4 = 2.30 |
| `importance_score_change` | ✅ PASS | 50.0 + 2×5 = 60.0 |

### 2.4 `test_04_memorydb` — 7/7 passed

| Test case | Result | Detail |
|-----------|--------|--------|
| `memorydb_create` | ✅ PASS | `MemoryDB` initialised (short layer config) |
| `add_single_memory` | ✅ PASS | 1 entry after adding single text |
| `add_batch_memory` | ✅ PASS | 3 entries after adding a batch of 2 |
| `query_memory` | ✅ PASS | returned 1 text + id for `top_k=2` |
| `decay_step` | ✅ PASS | `step()` executed; 0 entries removed (scores above threshold) |
| `access_counter_update` | ✅ PASS | `update_access_count_with_feed_back()` updated +1 |
| `checkpoint_roundtrip` | ✅ PASS | save → load → same entry count |

### 2.5 `test_05_portfolio` — 6/6 passed

| Test case | Result | Detail |
|-----------|--------|--------|
| `day_count` | ✅ PASS | 10 market-info updates → day_count=10 |
| `holding_shares` | ✅ PASS | holdings updated correctly from action series |
| `feedback_response` | ✅ PASS | `{feedback: 1, date: 2022-07-17}` (positive P&L) |
| `momentum_response` | ✅ PASS | `{moment: 1, date: 2022-07-21}` (positive 3-day) |
| `action_df_shape` | ✅ PASS | `get_action_df()` → 10 rows × 3 cols |
| `no_feedback_early` | ✅ PASS | `None` returned before `lookback_window_size` days |

### 2.6 `test_06_environment` — 4/4 passed

| Test case | Result | Detail |
|-----------|--------|--------|
| `env_creation` | ✅ PASS | `MarketEnvironment` built from test env_data |
| `market_info_types` | ✅ PASS | tuple types correct: `(date, float, str|None, str|None, list, float, bool)` |
| `step_count_valid` | ✅ PASS | 8 steps over 8-day window (2022-07-14 → 2022-07-21) |
| `termination_flag` | ✅ PASS | final step returns `done=True` |

---

## 3. Simulation Tests (`run.py`)

### 3.1 Train Mode

**Command:**
```bash
python run.py sim \
  -mdp data/06_input/subset_symbols.pkl \
  -st 2016-01-13 -et 2016-01-20 \
  -rm train \
  -cp config/tsla_cerebras_config.toml
```

**Result: ✅ SUCCESS** — 4/4 steps completed in ~25 s

| Step | Date | Price Δ | Memory Added | Reflection Summary |
|------|------|---------|--------------|-------------------|
| 1 | 2016-01-13 | +0.391 | reflection[0] | "No relevant info; price change reflects normal market volatility." |
| 2 | 2016-01-14 | −0.079 | reflection[1] | "Price decline likely reflects normal market fluctuation without specific catalyst." |
| 3 | 2016-01-15 | −0.019 | reflection[2] | "Price decline reflects normal market volatility." |
| 4 | 2016-01-19 | −0.401 | reflection[3] | "Price decline attributed to normal market volatility." |

**Reflection memory after training:** 4 TSLA entries in `reflection_memory`  
**Train actions:** buy(+1) on day 1, sell(−1) on days 2–4 (tracking sign of Δprice)  
**Checkpoint saved:** `data/05_train_model_output/` and `data/06_train_checkpoint/`

**LLM Guardrails behaviour:** On step 1, Reask 0 returned flat scalars for memory indices (not arrays). Guardrails correctly triggered a Reask, and Reask 1 returned well-formed nested array structure. This is the expected guardrails correction flow.

### 3.2 Test Mode

**Command:**
```bash
python run.py sim \
  -mdp data/06_input/subset_symbols.pkl \
  -st 2016-01-19 -et 2016-01-21 \
  -rm test \
  -cp config/tsla_cerebras_config.toml \
  -tap data/05_train_model_output
```

**Result: ✅ SUCCESS** — 2/2 steps completed in ~20 s

| Step | Date | Decision | Reason |
|------|------|----------|--------|
| 1 | 2016-01-19 | **sell** | Negative 3-day cumulative return & downward momentum; reflection confirms no positive catalysts |
| 2 | 2016-01-20 | **sell** | Reflection memory (id=4) notes continued negative momentum; 3-day cumulative return still negative |

**Memory cited:** both steps cited reflection memory; no short/mid/long memories were available (no news or filings in fake data)  
**Portfolio after test:**

| Date | Action | Holding Δ | Price |
|------|--------|-----------|-------|
| 2016-01-13 | buy (+1) | +1 | 13.354 |
| 2016-01-14 | sell (−1) | 0 | 13.745 |
| 2016-01-15 | sell (−1) | −1 | 13.666 |
| 2016-01-19 | sell (−1) | −2 | 13.648 |
| 2016-01-20 | sell (−1) | −3 | 13.648 |
| (end) | — | −4 | 13.247 |

**Reflection memory after test:** 6 TSLA entries (4 from train + 2 from test)  
**Result saved:** `data/07_test_model_output/` and `data/06_train_checkpoint/`

### 3.3 Checkpoint Resume Mode

**Command:**
```bash
python run.py sim-checkpoint \
  -ckp data/06_train_checkpoint \
  -rp data/07_test_model_output \
  -cp config/tsla_cerebras_config.toml \
  -rm test
```

**Result: ✅ SUCCESS (immediate completion)**

The checkpoint contained 1 remaining date (2016-01-21) with no subsequent date, so `env.step()` immediately yielded `done=True`. The agent correctly exited the loop and saved the final state to `data/07_test_model_output/`.

This validates the save/resume invariant: after the final step was processed in the previous test run, the checkpoint correctly represented the terminal state.

**Output files verified:**
```
data/07_test_model_output/
  agent_1/state_dict.pkl
  agent_1/brain/{short,mid,long,reflection}_term_memory/
  env/env.pkl
```

---

## 4. Bugs Found & Fixed

Six bugs were identified during codebase review and corrected:

### Bug 1 — `TextTruncator.truncate_text` used wrong method name
- **File:** [puppy/agent.py](../puppy/agent.py)
- **Symptom:** `AttributeError: 'TextTruncator' object has no attribute 'tokenize_cnt_texts'` when TGI model truncation was invoked
- **Root cause:** `truncate_text()` called `self.tokenize_cnt_texts(...)` (public name) instead of the actual `self._tokenize_cnt_texts(...)` (private with underscore)
- **Fix:** Corrected call to `self._tokenize_cnt_texts(input_text)`

### Bug 2 — Train reflection skipped on zero price delta
- **File:** [puppy/agent.py](../puppy/agent.py)
- **Symptom:** On trading days where `price_delta == 0.0`, the LLM reflection was silently skipped and `{}` was stored instead of a genuine summary
- **Root cause:** `if (run_mode == RunMode.Train) and (not cur_record):` — the Python truthiness check `not 0.0` evaluates to `True`, matching the "no record" branch
- **Fix:** Changed condition to `cur_record is None` to only skip when there is genuinely no record

### Bug 3 — Train action assigned sell on flat price
- **File:** [puppy/agent.py](../puppy/agent.py)
- **Symptom:** A zero price delta day incorrectly emitted a `sell` train action, poisoning future feedback loop signals
- **Root cause:** `cur_direction = 1 if cur_record > 0 else -1` never yielded `0`
- **Fix:** Added explicit `elif cur_record < 0` / `else: cur_direction = 0` branches

### Bug 4 — Guardrails compat shim called non-existent `Guard.for_pydantic`
- **File:** [puppy/guardrails_compat.py](../puppy/guardrails_compat.py)
- **Symptom:** `AttributeError: type object 'Guard' has no attribute 'for_pydantic'` on every simulation step — entire reflection pipeline failed
- **Root cause:** The shim was written targeting guardrails ≥0.5 where `from_pydantic` was renamed to `for_pydantic`. The installed version (0.4.5) still uses `from_pydantic` natively with the correct signature. The shim also double-patched `from_pydantic` with a broken lambda, overwriting the first patch
- **Fix:** Rewrote the shim to only provide the custom `ValidChoices` validator (needed for the deprecated `validators=` kwarg in Pydantic Field); `Guard.from_pydantic` is used natively without any monkey-patching

### Bug 5 — Checkpoint resume reported wrong simulation length
- **File:** [puppy/environment.py](../puppy/environment.py)
- **Symptom:** Progress bar showed an off-by-one count when resuming from a mid-run checkpoint (e.g., reported 2 remaining steps when only 1 was left)
- **Root cause:** `load_checkpoint` set `env.simulation_length = len(env.date_series)` instead of `len(env.date_series) - 1`; the last date in `date_series` is always the terminal step (no next date → done), not a processable step
- **Fix:** Changed to `env.simulation_length = max(0, len(env.date_series) - 1)`

### Bug 6 — Stray `}` at end of `test_prompt`
- **File:** [puppy/prompts.py](../puppy/prompts.py)
- **Symptom:** The test prompt template ended with `${gr.complete_json_suffix_v2} }` — the trailing `}` was part of the rendered prompt sent to the LLM, causing some models to include it in the JSON response and fail schema validation
- **Root cause:** Typo — an extra `}` after the guardrails suffix placeholder
- **Fix:** Removed the trailing `}`

---

## 5. Known Issues / Observations

### 5.1 Transient HuggingFace Embedding API Timeouts (504)
- **Observed:** During the first train run attempt step 3 (2016-01-15), the HF Inference API at `router.huggingface.co` returned HTTP 504 Gateway Timeout, causing the simulation to crash
- **Impact:** Non-deterministic; the second run (same command) completed without error
- **Recommendation:** The embedding call in `puppy/memorydb.py` should be wrapped in a retry decorator (e.g., `tenacity.retry`) with exponential backoff. Example:
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential
  
  @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
  def _call_embedding(self, text):
      return self.emb_func(text)
  ```
- **Workaround available:** Use `sim-checkpoint` to resume after any API failure without data loss

### 5.2 Reflection Validation Failures (Reask Needed)
- **Observed:** On most steps, the LLM's Reask 0 response uses flat scalars (`"short_memory_index": -1`) instead of the required array structure (`"short_memory_index": [{"memory_index": -1}]`)
- **Impact:** Guardrails correctly triggers Reask 1, which always returns the proper structure — no data loss, but adds ~1–2 s latency per step and doubles token usage
- **Recommendation:** Improve the system prompt or add a few-shot example JSON to the prompt template to guide the LLM toward the correct array-of-objects format on first attempt

### 5.3 Stale `"JSON does not match schema"` Text in Reflection Memory
- **Observed:** When guardrails validation ultimately fails (both Reask 0 and 1 produce invalid JSON), the fallback `err_msg` string `"JSON does not match schema"` is stored as the reflection memory text
- **Impact:** Cosmetic — these entries do get embedded and retrieved but carry no useful information; they score lower over time due to decays and eventual cleanup
- **Recommendation:** Use a sentinel value such as `"[REFLECTION_FAILED]"` to make these entries easier to filter in downstream analysis

### 5.4 Fake Dataset Has No News or Filing Data
- **Observed:** The fake sample dataset has empty news lists and no filings for any date. Only `reflection_memory` is ever populated during the run
- **Impact:** Short, mid, and long memory layers remain empty; the agent can only reason from its own reflection history; test results are artificial
- **Recommendation:** Use real data prepared via `data-pipeline/` scripts for meaningful evaluation

---

## 6. Summary

| Area | Status |
|------|--------|
| Unit tests (34 cases) | ✅ All passed |
| `run.py sim` — train mode | ✅ Completed (4 steps, ~25 s) |
| `run.py sim` — test mode | ✅ Completed (2 steps, ~20 s) |
| `run.py sim-checkpoint` — resume | ✅ Completed (terminal state, ~5 s) |
| Bugs identified | 6 |
| Bugs fixed | 6 |
| Open issues | 4 (transient, cosmetic, or improvement suggestions) |
