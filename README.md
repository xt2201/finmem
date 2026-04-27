# FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![arXiv](https://img.shields.io/badge/arXiv-2311.13743-b31b1b.svg)](https://arxiv.org/abs/2311.13743)

```text
"So we beat on, boats against the current, borne back ceaselessly into the past."
                                        -- F. Scott Fitzgerald: The Great Gatsby
```

This repository contains the Python implementation of the paper:
[FINMEM: A Performance-Enhanced Large Language Model Trading Agent with Layered Memory and Character Design](https://arxiv.org/abs/2311.13743) [[PDF]](https://arxiv.org/pdf/2311.13743.pdf)

## Citation

```bibtex
@misc{yu2023finmem,
      title={FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design},
      author={Yangyang Yu and Haohang Li and Zhi Chen and Yuechen Jiang and Yang Li and Denghui Zhang and Rong Liu and Jordan W. Suchow and Khaldoun Khashanah},
      year={2023},
      eprint={2311.13743},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```

## Overview

FinMem combines:

- Character/profile prompts for the agent
- Layered memory (short/mid/long/reflection)
- LLM-based reflection to generate daily trading decisions

In this repository, the runtime has been extended with:

- US and VN market support
- VN news ingestion and optional translation-before-VADER sentiment scoring
- Checkpoint resume flow (`sim-checkpoint`)
- RL baselines (`DQN`, `A2C`, `PPO`) and comparison plotting

Default embedding model in current config is:

- `sentence-transformers/all-MiniLM-L6-v2`

## Repository Layout

```text
finmem/
|- config/                # Runtime configuration templates
|- data/                  # Inputs, checkpoints, outputs
|- data-pipeline/         # Data build/eval/visualization scripts
|- docs/                  # Additional docs
|- figures/               # Generated plots
|- puppy/                 # Core FinMem agent/runtime code
|- rl/                    # RL baseline training/eval utilities
|- tests/                 # Regression and validation tests
|- run.py                 # Typer CLI entrypoint
|- run_paper_eval.sh      # Near-paper split helper (train+test)
|- run_cerebras.sh        # Simple Cerebras wrapper script
|- run_gemini.sh          # Simple Gemini-compatible wrapper script
|- run_tgi.sh             # Simple TGI-compatible wrapper script
|- scripts/               # Batch / isolated-paper / env-stack helpers
|- setup.sh               # One-command setup for sample data flow
|- pyproject.toml
|- README.md
```

## Prerequisites

- Python `>=3.10,<3.11`
- Bash-compatible shell for helper scripts (`Git Bash` or `WSL` on Windows)
- `uv` recommended for setup

## Quickstart

### 1) Create environment and install dependencies

Recommended (uses repository setup script):

```bash
bash setup.sh
```

Activate environment after setup:

- Linux/macOS/WSL:

```bash
source .venv/bin/activate
```

- Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

- PowerShell on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) Configure `.env`

Copy `.env.example` and set keys for your workflow.

Required by workflow:

- US market data build: `ALPACA_API_KEY`, `ALPACA_API_SECRET_KEY`, `SEC_KEY`
- VN market runs (enforced): `CEREBRAS_API_KEY`
- Embedding/translation model download (when needed): `HF_TOKEN`
- Gemini/TGI/OpenAI-compatible wrappers: `OPENAI_API_KEY` (depending on endpoint)

Core runtime overrides:

| Variable | Purpose |
| --- | --- |
| `FINMEM_TRADING_SYMBOL` | Symbol fallback when config has empty `general.trading_symbol` |
| `FINMEM_MARKET_MODE` / `FINMEM_MARKET` | Market selection (`US` or `VN`) |
| `FINMEM_CONFIG_PATH` | Override config TOML path |
| `FINMEM_MARKET_DATA_PATH` | Override market input pickle path |
| `FINMEM_CHECKPOINT_PATH` | Override checkpoint directory |
| `FINMEM_RESULT_PATH` | Override result directory |
| `FINMEM_TRAINED_AGENT_PATH` | Train artifact path for test mode |

VN pipeline and sentiment overrides:

| Variable | Purpose |
| --- | --- |
| `FINMEM_VNSTOCK_SOURCE` | VN source selector (KBS-only is enforced) |
| `FINMEM_VNSTOCK_NEWS_LIMIT` | Total VN news rows to fetch |
| `FINMEM_VNSTOCK_NEWS_PAGE_SIZE` | VN page size (capped for KBS) |
| `FINMEM_VNSTOCK_NEWS_MAX_PAGE` | Max VN pages to request |
| `FINMEM_VN_NEWS_ALIGN_WINDOW_DAYS` | Alignment window from news date to trading date |
| `FINMEM_VN_TRANSLATE_FOR_VADER` | Enable translation-before-VADER (`1`/`0`) |
| `FINMEM_VN_TRANSLATION_MODEL` | Translation model (default `Helsinki-NLP/opus-mt-vi-en`) |
| `FINMEM_VN_TRANSLATION_LOCAL_ONLY` | Use local cache only for translation model |
| `FINMEM_VN_TRANSLATION_MAX_LENGTH` | Translation truncation length |

Build/eval helper overrides:

| Variable | Purpose |
| --- | --- |
| `FINMEM_BUILD_START`, `FINMEM_BUILD_END` | Defaults for data build script |
| `FINMEM_MAX_NEWS_PER_DAY`, `FINMEM_NEWS_SLEEP_SECONDS` | News build controls |
| `FINMEM_EVAL_START`, `FINMEM_EVAL_END` | Defaults for metrics/plot scripts |
| `FINMEM_STATE_DICT_PATH` | Default FinMem state dict path for eval scripts |

## Full end-to-end run: six tickers (US + VN)

This repo includes scripts so a clean clone can run the **same near-paper pipeline as** `run_paper_eval.sh` (train → test → `sim-rl` → 5-measure metrics → plot) for:

| Ticker | Market | Input date range (build `09_build_paper_input.py`) | Train / test split (eval) |
| --- | --- | --- | --- |
| NFLX, AMZN, MSFT | US | `2021-08-17` – `2023-04-10` | Paper: train `2021-08-17`–`2022-10-05`, test `2022-10-06`–`2023-04-10` |
| BID | VN | `2025-04-01` – `2025-05-31` | Calendar months: train `2025-04`, test `2025-05` (trading days from pickle) |
| MBB | VN | `2025-07-01` – `2025-08-31` | Train `2025-07`, test `2025-08` |
| FPT | VN | `2025-05-01` – `2025-06-30` | Train `2025-05`, test `2025-06` |

VN builds should use KBS + translation before VADER (see table above): `FINMEM_VNSTOCK_SOURCE=KBS`, `FINMEM_VN_TRANSLATE_FOR_VADER=1`.

**Requirements:** `.env` filled from `.env.example` (at minimum `ALPACA_*`, `SEC_KEY` for US data build; `CEREBRAS_API_KEY` for all LLM sims in this config; `HF_TOKEN` if you need to download translation/embedding models). **Do not commit real keys.** Long runs (often **many hours per US symbol**) should use a stable session: `nohup`, `tmux`/`screen`, and on macOS `caffeinate` so the machine does not sleep.

**One script — build (if needed) and run all six in order** (isolated output dirs: `data/09_results_nflx/`, `data/09_results_amzn/`, … so runs do not overwrite each other):

```bash
source .venv/bin/activate
# Optional: set -a; source scripts/source_env_stack.sh .env .env.alternate; set +a
bash scripts/run_readme_batch_all.sh
```

Environment flags for that script:

- `SKIP_BUILD=1` — only run eval; expects pickles under `data/03_model_input/<ticker>.pkl` already.
- `FORCE_REBUILD=1` — always re-run `09_build_paper_input.py` (slower, uses APIs).

**Resume after an interrupted US train (NFLX) and finish the rest of the queue** — completes NFLX (checkpoint → test → RL → metrics), then runs AMZN, MSFT, and the three VN symbols:

```bash
# macOS: keeps the system awake while the outer shell runs (plugged in)
caffeinate -dims bash scripts/run_continue_to_completion.sh
```

**Single symbol, paper dates from env** (US or VN month pair auto-filled for VN):

```bash
SYMBOL=AMZN MARKET_MODE=US bash scripts/run_isolated_paper_eval.sh
export FINMEM_VN_TRANSLATE_FOR_VADER=1
SYMBOL=BID MARKET_MODE=VN VN_TRAIN_MONTH=2025-04 VN_TEST_MONTH=2025-05 bash scripts/run_isolated_paper_eval.sh
```

**Inspect VN train/test dates derived from a pickle** (optional):

```bash
python scripts/vn_train_test_from_pkl.py data/03_model_input/bid.pkl --train-month 2025-04 --test-month 2025-05
```

**Stack multiple env files** (e.g. rotate API keys) without bash-unfriendly `.env` spacing issues:

```bash
set -a; source scripts/source_env_stack.sh .env .env.backup; set +a
```

## Build Market Input Data

### US market

```bash
python data-pipeline/09_build_paper_input.py \
  --market US \
  --symbol TSLA \
  --start 2021-08-17 \
  --end 2023-04-10
```

### VN market

```bash
export FINMEM_MARKET_MODE=VN
export FINMEM_VNSTOCK_SOURCE=KBS
export FINMEM_VN_TRANSLATE_FOR_VADER=1

python data-pipeline/09_build_paper_input.py \
  --market VN \
  --symbol VCI \
  --start 2024-01-02 \
  --end 2024-12-31
```

Default output path is `data/03_model_input/<symbol>.pkl`.

## Run FinMem Simulation

### Symbol resolution precedence

Runtime resolves trading symbol in this order:

1. `general.trading_symbol` in config
2. `FINMEM_TRADING_SYMBOL`
3. CLI `--trading-symbol`
4. Internal default `TSLA`

### Train mode

```bash
python run.py sim \
  -mdp data/03_model_input/tsla.pkl \
  -st 2021-08-17 \
  -et 2022-10-05 \
  -rm train \
  -cp config/finmem_cerebras_config.toml \
  -ckp data/06_train_checkpoint \
  -rp data/05_train_model_output
```

### Test mode

```bash
python run.py sim \
  -mdp data/03_model_input/tsla.pkl \
  -st 2022-10-06 \
  -et 2023-04-10 \
  -rm test \
  -cp config/finmem_cerebras_config.toml \
  -tap data/05_train_model_output \
  -ckp data/08_test_checkpoint \
  -rp data/09_results
```

### Resume from checkpoint

```bash
python run.py sim-checkpoint \
  -cp config/finmem_cerebras_config.toml \
  -rm train \
  -ckp data/06_train_checkpoint \
  -rp data/05_train_model_output \
  --trading-symbol TSLA
```

Pass `--trading-symbol` (or set `FINMEM_TRADING_SYMBOL`) so the checkpoint symbol matches the pickle (e.g. NFLX, not the default `TSLA`).

### Near-paper split helper

`run_paper_eval.sh` now runs the full default pipeline:

- Train: `2021-08-17` to `2022-10-05`
- Test: `2022-10-06` to `2023-04-10`
- RL baselines with retry (`DQN`, `A2C`, `PPO`)
- Final 5-measure metrics (`FinMem`, `Buy & Hold`, `A2C`, `DQN`, `PPO`)
- Final 5-measure cumulative return figure

```bash
export SYMBOL=TSLA
export MARKET_MODE=US
bash run_paper_eval.sh
```

VN example:

```bash
export SYMBOL=VCI
export MARKET_MODE=VN
export CONFIG_PATH=config/finmem_cerebras_vn_config.toml
bash run_paper_eval.sh
```

## Evaluate and Visualize

Compute metrics for 5 measures (`FinMem`, `Buy & Hold`, `A2C`, `DQN`, `PPO`):

```bash
python data-pipeline/07-metrics.py \
  --market US \
  --ticker TSLA \
  --start 2022-10-06 \
  --end 2023-04-10 \
  --market-data-path data/03_model_input/tsla.pkl \
  --state-dict-path data/09_results/agent_1/state_dict.pkl \
  --actions-output-dir data/09_results \
  --save-path data/09_results/TSLA_metrics_5measures.csv
```

Plot cumulative return comparison with 5 curves:

```bash
python data-pipeline/06-Visualize-results.py \
  --market US \
  --ticker TSLA \
  --start 2022-10-06 \
  --end 2023-04-10 \
  --market-data-path data/03_model_input/tsla.pkl \
  --state-dict-path data/09_results/agent_1/state_dict.pkl \
  --actions-output-dir data/09_results \
  --save-path data/09_results/TSLA_5measures.png
```

## RL Baselines and Comparison Plot

Train RL baselines and generate comparison plot:

```bash
python run.py sim-rl \
  --algorithm all \
  --market-data-path data/03_model_input/tsla.pkl \
  --train-start 2021-08-17 \
  --train-end 2022-10-05 \
  --test-start 2022-10-06 \
  --test-end 2023-04-10 \
  --episodes 20 \
  --window 10 \
  --market-mode US \
  --seed 42 \
  --retry-count 2 \
  --retry-seed-step 101 \
  --finmem-state-dict data/09_results/agent_1/state_dict.pkl \
  --actions-output-dir data/09_results
```

Saved RL action artifacts:

- `data/09_results/<TICKER>_actions_dqn.pkl`
- `data/09_results/<TICKER>_actions_a2c.pkl`
- `data/09_results/<TICKER>_actions_ppo.pkl`

## Script Shortcuts

- `run_cerebras.sh`: quick train/test wrapper for Cerebras config
- `run_gemini.sh`: quick wrapper using Gemini-style config
- `run_tgi.sh`: quick wrapper for TGI endpoint config
- `run_paper_eval.sh`: canonical full helper (train + test + RL + 5-measure metrics/plot)
- `scripts/run_readme_batch_all.sh`: build (optional) + isolated paper eval for NFLX, AMZN, MSFT, BID, MBB, FPT
- `scripts/run_isolated_paper_eval.sh`: same as `run_paper_eval.sh` with per-ticker `data/09_results_<ticker>/` (and VN month envs)
- `scripts/run_continue_to_completion.sh`: resume NFLX train from `data/06_train_checkpoint_nflx/`, then the remaining tickers
- `scripts/vn_train_test_from_pkl.py`: print or export train/test dates for VN month splits
- `scripts/source_env_stack.sh`: load multiple dotenv files in order (later overrides)

## Artifact Lifecycle and Cleanup

Common generated directories:

| Directory | Content | Keep Policy |
| --- | --- | --- |
| `data/03_model_input/` | Built env_data pickles | Keep (source for reruns) |
| `data/04_model_output_log/` | Runtime logs | Optional keep |
| `data/05_train_model_output/` | Train outputs (`state_dict.pkl`) | Keep latest stable |
| `data/06_train_checkpoint/` | Train checkpoints | Optional after successful train |
| `data/08_test_checkpoint/` | Test checkpoints | Optional after successful test |
| `data/09_results/` | Test outputs (`state_dict.pkl`) | Keep latest stable |
| `figures/` | Generated plots | Keep canonical plots only |

Minimal keep set for reproducibility:

- Input market pickle in `data/03_model_input/`
- Train `state_dict.pkl` used by test
- Test `state_dict.pkl` used by metrics/plots
- Final report plots and metrics CSV

## Troubleshooting

- `Resolved trading_symbol 'X' not found in market data`:
  symbol is missing in selected `env_data` range. Rebuild input or pass correct symbol.

- `start_date and end_date must be present in market data`:
  `-st` and `-et` must match exact trading-day keys in the pickle, not arbitrary calendar dates.

- `VN mode requires Cerebras provider only`:
  use VN Cerebras config and set `CEREBRAS_API_KEY`.

- `trained_agent_path is required in test mode`:
  pass `-tap` pointing to directory that contains `agent_1/state_dict.pkl`.

- Action/price horizon mismatch in metrics/plot:
  state dict, RL action artifacts, and requested date window must match exactly.
  Re-run test + sim-rl with the same `--test-start/--test-end` window.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pipiku915/FinMem-LLM-StockTrading&type=Date)](https://star-history.com/#pipiku915/FinMem-LLM-StockTrading&Date)
