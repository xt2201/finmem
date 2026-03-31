# FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![arXiv](https://img.shields.io/badge/arXiv-2311.13743-b31b1b.svg)](https://arxiv.org/abs/2311.13743)

```text
"So we beat on, boats against the current, borne back ceaselessly into the past."
                                        -- F. Scott Fitzgerald: The Great Gatsby
```

This repo provides the Python source code for the paper:
[FINMEM: A Performance-Enhanced Large Language Model Trading Agent with Layered Memory and Character Design](https://arxiv.org/abs/2311.13743) [[PDF]](https://arxiv.org/pdf/2311.13743.pdf)

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
Update (Date: 01-16-2024)

Our work, "FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design," has been selected for an extended abstract at the AAAI Spring Symposium on Human-Like Learning!

Update (Date: 03-11-2024)

Our paper, "FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design", has been accepted by ICLR Workshop LLM Agents!

Update (Date: 06-16-2024)

Thank you to all the participants and organizers of the IJCAI2024 challenge, "Financial Challenges in Large Language Models - FinLLM". Our team, FinMem, was thrilled to contribute to Task 3: Single Stock Trading.

As the challenge wrapped up yesterday (06/15/2024), we reflect on the innovative approaches and insights gained throughout this journey. A total of 12 teams participated, each bringing unique perspectives and solutions to the forefront of financial AI and Large Language Models.

We invite the community to continue engaging with us as we look forward to further developments and collaborations in this exciting field.


Recent advancements in Large Language Models (LLMs) have exhibited notable efficacy in question-answering (QA) tasks across diverse domains. Their prowess in integrating extensive web knowledge has fueled interest in developing LLM-based autonomous agents. While LLMs are efficient in decoding human instructions and deriving solutions by holistically processing historical inputs, transitioning to purpose-driven agents requires a supplementary rational architecture to process multi-source information, establish reasoning chains, and prioritize critical tasks. Addressing this, we introduce FinMem, a novel LLM-based agent framework devised for financial decision-making, encompassing three core modules: Profiling, to outline the agent's characteristics; Memory, with layered processing, to aid the agent in assimilating realistic hierarchical financial data; and Decision-making, to convert insights gained from memories into investment decisions. Notably, FinMem's memory module aligns closely with the cognitive structure of human traders, offering robust interpretability and real-time tuning. Its adjustable cognitive span allows for the retention of critical information beyond human perceptual limits, thereby enhancing trading outcomes. This framework enables the agent to self-evolve its professional knowledge, react agilely to new investment cues, and continuously refine trading decisions in the volatile financial environment. We first compare FinMem with various algorithmic agents on a scalable real-world financial dataset, underscoring its leading trading performance in stocks and funds. We then fine-tuned the agent's perceptual spans to achieve a significant trading performance. Collectively, FinMem presents a cutting-edge LLM agent framework for automated trading, boosting cumulative investment returns.

![1](figures/memory_flow.png)
![2](figures/workflow.png)
![3](figures/character.png)

## Repository Structure

```bash
finmem
|-- LICENSE
|-- README.md
|-- config           # Configurations for the program
|-- data             # Data
|-- puppy            # Source code
|-- run.py           # Entry point of the program
|-- tests            # Unit and integration tests
|-- run_examples.sh  # Script for running examples
```



## Usage

### Setting Environment Variables

The project uses Cerebras for LLM generation and HuggingFace for embeddings (defaulting to `intfloat/multilingual-e5-large`).

Create a `.env` file in the root directory and ensure all API keys below are provided:

```bash
CEREBRAS_API_KEY="<Your Cerebras API Key>"
HF_TOKEN="<Your HF token>"
SEC_KEY="<Your SEC API key (from sec-api.io)>"
ALPACA_API_KEY="<Your Alpaca API key>"
ALPACA_API_SECRET_KEY="<Your Alpaca API secret key>"
ALPACA_ENDPOINT="https://paper-api.alpaca.markets/v2"
ALPACA_NEWS_ENDPOINT="https://data.alpaca.markets/v1beta1/news"
```

### Quick Setup

The recommended way to set up the environment and extract historical data is using the provided setup script. This script requires [uv](https://github.com/astral-sh/uv).

Run the following command to automatically create a virtual environment, install dependencies, and format the compressed sample dataset for the simulation:
```bash
bash setup.sh
```

Once the setup is complete, you can enter the environment using:
```bash
source .venv/bin/activate
```

### Running the Paper Evaluation Pipeline (Any Symbol)

You can run the paper-style flow for any supported ticker symbol by selecting the symbol through config or environment variables.

Precedence used by runtime symbol resolution is:

1. `general.trading_symbol` in config
2. `FINMEM_TRADING_SYMBOL` environment variable
3. CLI `--trading-symbol`
4. Internal default (`TSLA`)

**Note:** The compiled dataset `data/03_model_input/tsla.pkl` is included in this repository for quick testing.

1. **(Optional) Build the Realistic Dataset:** Fetch real-time data from `yfinance`, download news coverage securely from `Alpaca`, and pull standard `10-K` / `10-Q` company filings using the SEC API. Ensure your `.env` contains the keys listed above.

```bash
python data-pipeline/09_build_paper_input.py --symbol AAPL --start 2021-08-17 --end 2023-04-10
```
*(This generates `data/03_model_input/aapl.pkl` if no `--output-path` is provided.)*

2. **Run the Full Paper Simulation Timeline:** Train memory representations from `2021-08-17` to `2022-10-05` and perform test trades from `2022-10-06` to `2023-04-10`.

```bash
export SYMBOL=AAPL
bash run_paper_eval.sh
```

*(Note that `run_paper_eval.sh` already establishes MacOS stability flags like `KMP_DUPLICATE_LIB_OK="TRUE"` and `OMP_NUM_THREADS="1"` for smooth execution.)*

3. **Evaluate and Visualize Results:**
After the test is completed, calculate financial metrics (Sharpe Ratio, Cumulative Return, MDD) and visualize performance against baseline strategies (Buy & Hold):

```bash
python data-pipeline/07-metrics.py
python data-pipeline/06-Visualize-results.py
```

### Optional: Use a Custom OpenAI-Compatible Endpoint (ngrok/vLLM)

If you want to run FinMem against a self-hosted OpenAI-compatible server (for example a vLLM endpoint exposed by ngrok), use the provided config:

```bash
export FINMEM_TRADING_SYMBOL=AAPL
python run.py sim -mdp data/03_model_input/aapl.pkl -st 2021-08-17 -et 2022-10-05 -rm train -cp config/finmem_openai_compatible_config.toml
```

Notes:
- Set `[chat].end_point` to your server base URL (for example `https://.../v1`) or full chat endpoint (`https://.../v1/chat/completions`).
- Set `[chat].model` to your hosted model ID.
- Set `[chat].openai_compatible = true`.
- Use `[chat].api_key = "EMPTY"` if your server does not require a real key.
- Use `config/finmem_cerebras_config.toml`, `config/finmem_openai_compatible_config.toml`, `config/finmem_gemini_config.toml`, or `config/finmem_tgi_config.toml` as generic templates.

## Program Usage

The program uses two modes: `train` and `test`. 
- **Train mode**: Streams information over time to populate the agent's memory index without generating trades.
- **Test mode**: The agent queries its existing memory databases against new daily information to make actual trading choices.

### Example: Running a Practical Simulation

Ensure OpenMP library duplication is allowed for Huggingface tokenizers prior to running:
```bash
export KMP_DUPLICATE_LIB_OK="TRUE"
```

1. **Populate Memory (Train Mode)** spanning Jan 13 -> Jan 15:
```bash
python run.py sim -mdp data/06_input/subset_symbols.pkl -st 2016-01-13 -et 2016-01-15 -rm train -cp config/finmem_cerebras_config.toml
```

2. **Generate Trades (Test Mode)** for Jan 19 -> Jan 20 utilizing the previously generated checkpoints:
```bash
python run.py sim -mdp data/06_input/subset_symbols.pkl -st 2016-01-19 -et 2016-01-20 -rm test -cp config/finmem_cerebras_config.toml -tap data/05_train_model_output
```

### Advanced Usage Information

```bash
 Usage: run.py sim [OPTIONS]

 Start Simulation

 Options
 --market-data-path    -mdp      TEXT  The environment data pickle path [default: data/03_model_input/<symbol>.pkl]
 --start-time          -st       TEXT  The training or test start time [default: 2022-08-16]
 --end-time            -et       TEXT  The training or test end time [default: 2022-10-04]
 --run-model           -rm       TEXT  Run mode: train or test [default: train]
 --config-path         -cp       TEXT  config file path [default: config/finmem_cerebras_config.toml]
 --checkpoint-path     -ckp      TEXT  The checkpoint save path [default: data/06_train_checkpoint]
 --result-path         -rp       TEXT  The result save path [default: data/05_train_model_output]
 --trained-agent-path  -tap      TEXT  Only used in test mode, the path of trained agent [default: None]
 --trading-symbol      -sym      TEXT  Optional symbol override used only when config has no general.trading_symbol
 --help                                Show this message and exit.
```

If the API disconnects or rate-limits, the training process can be resumed securely from checkpoints:

```bash
 Usage: run.py sim-checkpoint [OPTIONS]

 Start Simulation from checkpoint

 Options
 --checkpoint-path  -ckp     TEXT  The checkpoint path [default: data/06_train_checkpoint]
 --result-path      -rp      TEXT  The result save path [default: data/05_train_model_output]
 --config-path      -cp      TEXT  config file path [default: config/finmem_cerebras_config.toml]
 --run-model        -rm      TEXT  Run mode: train or test [default: train]
 --trading-symbol   -sym     TEXT  Optional symbol override used only when config has no general.trading_symbol
 --help                            Show this message and exit.
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pipiku915/FinMem-LLM-StockTrading&type=Date)](https://star-history.com/#pipiku915/FinMem-LLM-StockTrading&Date)
