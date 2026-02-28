# FinMem Usecase Evaluation Report

We evaluated the FinMem-LLM-StockTrading repository using two Cerebras hosted models: `llama3.1-8b` and `zai-glm-4.7`.
Both **Train** and **Test** simulation modes were run successfully on a mock TSLA dataset to observe the memory mechanisms and LLM trading decisions.

## 1. Execution Pipeline Analysis
The `run.py` script and the inner loop processing (`puppy/agent.py` and `puppy/environment.py`) function as designed.
- **Data Loading**: Parses mock financial/news pickle dictionaries correctly.
- **Model Storage**: Memory checkpoints update efficiently and accurately segment into short, mid, long, and reflection memory tiers based on decay and importance scores.

## 2. Model Evaluation (`llama3.1-8b` vs `zai-glm-4.7`)

### A. Context and Memory Processing
Both models successfully navigated the complex JSON environment provided by the codebase. They can generate memories and correctly rank the most recent/important contextual snippets (e.g. tracking "Market is stable" vs "Tesla announced good earnings" vs "New factory opens").

- `zai-glm-4.7` was notably faster during the sequential LLM calls than `llama3.1-8b`.

### B. Output Quality & JSON Validation Issues
The repository relies heavily on the `guardrails-ai` library to enforce strict JSON schemas for LLM responses. Our empirical tests uncovered significant LLM performance degradation and hallucination during these JSON coercion steps (the "Reask" phase).

#### `llama3.1-8b` Hallucination
When `llama3.1-8b` analyzed positive news (Earnings and New Factory), its RAW generation natively output:
`"investment_decision": "buy"`

However, when guardrails attempted to validate and re-ask the LLM for a sturdier schema structure, the smaller `llama3.1-8b` model flipped its logic, rewriting the output as `sell` while keeping the exact same positive `"summary_reason"`.

#### `zai-glm-4.7` Hallucination
`zai-glm-4.7` suffered from instructions format confusion during the Guardrails re-ask step.
Its RAW generation perfectly captured the scenario:
`"investment_decision": "buy"`
`"summary_reason": "The positive news about the opening of a new factory (4) boosts investor sentiment, encouraging buying activity and pushing stock prices higher..."`

However, during the Reask 1 phase, the model confused the "desired format instructions" with the output text itself, generating placeholder types:
`"investment_decision": "string"`
`"summary_reason": "string"`

This caused the active simulation to attempt to perform a trade action named `"string"`.

## Conclusion
The FinMem layered-memory architecture works solidly in managing continuous financial data flows. However, using smaller (< 70B parameter) or highly-tuned LLMs like `llama3.1-8b` or `zai-glm-4.7` with zero-shot rigid JSON verifications (such as `guardrails`) can lead to unintended prompt confusion. 

For production use cases on smaller models, it is recommended to either:
1. Simplify the required JSON outputs to reduce cognitive load.
2. Rely purely on regex/Python logic to parse out the `buy/sell/hold` markers instead of complex LLM validation loops.
