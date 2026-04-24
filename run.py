import os
import toml
import typer
import logging
import pickle
import warnings
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from puppy import MarketEnvironment, LLMAgent, RunMode
from rl import (
    TradingEnv,
    DQNAgent,
    A2CAgent,
    PPOAgent,
    train_dqn,
    train_a2c,
    train_ppo,
    evaluate_agent,
    load_env_data,
    set_global_seed,
    extract_price_series,
    action_indices_to_positions,
    align_actions_to_price_horizon,
    actions_to_cum_returns,
    load_finmem_actions_aligned,
    plot_cumulative_returns,
)
from rl.env import TradingEnvConfig
from rl.algos import DQNConfig, A2CConfig, PPOConfig
from puppy.runtime_config import (
    DEFAULT_MARKET_MODE,
    DEFAULT_TRADING_SYMBOL,
    expand_symbol_template,
    resolve_market_mode,
    resolve_path,
    resolve_trading_symbol,
    validate_symbol_in_market_data,
)


def _strip_wrapped_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_dotenv_compat() -> None:
    load_dotenv(override=False)
    env_candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.getcwd(), "..", ".env"),
    ]
    for env_path in env_candidates:
        if not os.path.exists(env_path):
            continue
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = _strip_wrapped_quotes(value.strip())
                if key and key not in os.environ:
                    os.environ[key] = value


# set up
_load_dotenv_compat()
app = typer.Typer(name="puppy")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG_PATH = os.path.join("config", "finmem_cerebras_config.toml")


def _default_market_data_path(symbol: str) -> str:
    return os.path.join("data", "03_model_input", f"{symbol.lower()}.pkl")


def _ensure_cerebras_provider_for_vn(
    config: Dict[str, Any],
    market_mode: str,
    config_path: str,
) -> None:
    if market_mode != "VN":
        return

    chat = config.get("chat", {}) if isinstance(config, dict) else {}
    if not isinstance(chat, dict):
        raise ValueError(
            f"Invalid chat config in {config_path}. VN mode requires a valid Cerebras chat configuration."
        )

    model = str(chat.get("model", "")).strip().lower()
    endpoint = str(chat.get("end_point", "")).strip().lower()
    openai_compatible = bool(chat.get("openai_compatible", False))

    if openai_compatible or model.startswith("gemini-pro") or model.startswith("tgi"):
        raise ValueError(
            "VN mode requires Cerebras provider only. "
            "Use config/finmem_cerebras_vn_config.toml or another Cerebras config "
            "(openai_compatible=false, model not gemini-pro/tgi)."
        )

    if endpoint and "api.cerebras.ai" not in endpoint:
        raise ValueError(
            "VN mode requires a Cerebras endpoint (https://api.cerebras.ai/...). "
            f"Current endpoint in {config_path}: {chat.get('end_point')}"
        )

    if not os.environ.get("CEREBRAS_API_KEY"):
        raise ValueError(
            "VN mode with Cerebras provider requires CEREBRAS_API_KEY in environment."
        )


def _prepare_common_runtime(
    config_path: Optional[str],
    checkpoint_path: Optional[str],
    result_path: Optional[str],
    trading_symbol: Optional[str],
    market_mode: Optional[str],
) -> Tuple[str, str, str, Dict[str, Any], str, str]:
    resolved_config_path = resolve_path(
        config_path,
        "FINMEM_CONFIG_PATH",
        DEFAULT_CONFIG_PATH,
    )
    resolved_checkpoint_path = resolve_path(
        checkpoint_path,
        "FINMEM_CHECKPOINT_PATH",
        os.path.join("data", "06_train_checkpoint"),
    )
    resolved_result_path = resolve_path(
        result_path,
        "FINMEM_RESULT_PATH",
        os.path.join("data", "05_train_model_output"),
    )

    if not os.path.exists(resolved_config_path):
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    config = toml.load(resolved_config_path)
    resolved_symbol = resolve_trading_symbol(
        config=config,
        cli_symbol=trading_symbol,
        default_symbol=DEFAULT_TRADING_SYMBOL,
    )
    resolved_market_mode = resolve_market_mode(
        config=config,
        cli_market_mode=market_mode,
        default_market_mode=DEFAULT_MARKET_MODE,
    )
    config.setdefault("general", {})["trading_symbol"] = resolved_symbol
    config["general"]["market_mode"] = resolved_market_mode
    if "character_string" in config["general"]:
        config["general"]["character_string"] = expand_symbol_template(
            config["general"]["character_string"],
            resolved_symbol,
        )
    _ensure_cerebras_provider_for_vn(
        config=config,
        market_mode=resolved_market_mode,
        config_path=resolved_config_path,
    )

    return (
        resolved_config_path,
        resolved_checkpoint_path,
        resolved_result_path,
        config,
        resolved_symbol,
        resolved_market_mode,
    )


def _configure_run_logger(symbol: str) -> logging.Logger:
    os.makedirs(os.path.join("data", "04_model_output_log"), exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_path = os.path.join(
        "data",
        "04_model_output_log",
        f"{symbol}_run.log",
    )
    has_same_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(getattr(handler, "baseFilename", ""))
        == os.path.abspath(log_path)
        for handler in logger.handlers
    )
    if not has_same_file_handler:
        file_handler = logging.FileHandler(
            log_path,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(logging_formatter)
        logger.addHandler(file_handler)
    return logger


def _parse_run_mode(run_mode: str) -> RunMode:
    if run_mode == "train":
        return RunMode.Train
    if run_mode == "test":
        return RunMode.Test
    raise ValueError("Run mode must be train or test")


def _resolve_actions_output_dir(actions_output_dir: Optional[str]) -> str:
    default_dir = os.environ.get("FINMEM_RESULT_PATH", os.path.join("data", "09_results"))
    return resolve_path(
        actions_output_dir,
        "FINMEM_RL_ACTIONS_OUTPUT_DIR",
        default_dir,
    )


def _save_rl_actions(
    output_dir: str,
    symbol: str,
    algorithm: str,
    market_mode: str,
    start: str,
    end: str,
    actions: list[int],
    seed: int,
    attempt: int,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_actions_{algorithm.lower()}.pkl")
    payload = {
        "algorithm": algorithm,
        "symbol": symbol,
        "market_mode": market_mode,
        "start": start,
        "end": end,
        "seed": int(seed),
        "attempt": int(attempt),
        "actions": [int(a) for a in actions],
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def _run_rl_baseline_with_retry(
    algorithm: str,
    train_dates,
    train_prices,
    test_dates,
    test_prices,
    window: int,
    transaction_cost: float,
    episodes: int,
    seed: int,
    retry_count: int,
    retry_seed_step: int,
) -> Tuple[list[int], list[float], int, int]:
    last_error = None
    total_attempts = max(1, int(retry_count) + 1)

    for attempt_idx in range(total_attempts):
        attempt = attempt_idx + 1
        attempt_seed = int(seed) + attempt_idx * int(retry_seed_step)
        set_global_seed(attempt_seed)
        env_config = TradingEnvConfig(window=window, transaction_cost=transaction_cost)
        train_env = TradingEnv(prices=train_prices, dates=train_dates, config=env_config)
        test_env = TradingEnv(prices=test_prices, dates=test_dates, config=env_config)

        try:
            if algorithm == "DQN":
                agent = DQNAgent(train_env.obs_size, train_env.action_size, DQNConfig())
                train_dqn(train_env, agent, episodes=episodes)
            elif algorithm == "A2C":
                agent = A2CAgent(train_env.obs_size, train_env.action_size, A2CConfig())
                train_a2c(train_env, agent, episodes=episodes)
            elif algorithm == "PPO":
                agent = PPOAgent(train_env.obs_size, train_env.action_size, PPOConfig())
                train_ppo(train_env, agent, episodes=episodes)
            else:
                raise ValueError("algorithm must be one of: DQN, A2C, PPO")

            algo_actions = evaluate_agent(test_env, agent, deterministic=True)
            algo_positions = action_indices_to_positions(algo_actions)
            aligned_positions = align_actions_to_price_horizon(
                actions=algo_positions,
                prices_len=len(test_prices),
                window=window,
            )
            expected_horizon = max(0, len(test_prices) - 1)
            if len(aligned_positions) != expected_horizon:
                raise ValueError(
                    f"{algorithm} aligned action length mismatch: expected {expected_horizon}, got {len(aligned_positions)}"
                )

            returns = actions_to_cum_returns(test_prices.tolist(), aligned_positions)
            if not bool(np.isfinite(np.asarray(returns, dtype=np.float64)).all()):
                raise ValueError(f"{algorithm} produced non-finite cumulative returns")

            return aligned_positions, returns, attempt_seed, attempt
        except Exception as exc:
            last_error = exc
            typer.secho(
                f"{algorithm} attempt {attempt}/{total_attempts} failed with seed {attempt_seed}: {exc}",
                fg=typer.colors.YELLOW,
            )

    raise RuntimeError(
        f"{algorithm} failed after {total_attempts} attempts (base_seed={seed}, retry_seed_step={retry_seed_step})."
    ) from last_error


@app.command("sim", help="Start Simulation", rich_help_panel="Simulation")
def sim_func(
    market_data_info_path: Union[str, None] = typer.Option(
        None,
        "-mdp",
        "--market-data-path",
        help="The environment data pickle path",
    ),
    start_time: str = typer.Option(
        "2022-08-16", "-st", "--start-time", help="The start time"
    ),
    end_time: str = typer.Option(
        "2022-10-04", "-et", "--end-time", help="The end time"
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
    config_path: Union[str, None] = typer.Option(
        None,
        "-cp",
        "--config-path",
        help="config file path",
    ),
    checkpoint_path: Union[str, None] = typer.Option(
        None,
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: Union[str, None] = typer.Option(
        None,
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    trained_agent_path: Union[str, None] = typer.Option(
        None,
        "-tap",
        "--trained-agent-path",
        help="Only used in test mode, the path of trained agent",
    ),
    trading_symbol: Union[str, None] = typer.Option(
        None,
        "-sym",
        "--trading-symbol",
        help="Optional symbol override used only when config does not set general.trading_symbol",
    ),
    market_mode: Union[str, None] = typer.Option(
        None,
        "-mm",
        "--market-mode",
        help="Market mode override (US or VN). Defaults to FINMEM_MARKET_MODE/FINMEM_MARKET when provided.",
    ),
) -> None:
    (
        config_path,
        checkpoint_path,
        result_path,
        config,
        resolved_symbol,
        resolved_market_mode,
    ) = _prepare_common_runtime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        trading_symbol=trading_symbol,
        market_mode=market_mode,
    )

    market_data_info_path = resolve_path(
        market_data_info_path,
        "FINMEM_MARKET_DATA_PATH",
        _default_market_data_path(resolved_symbol),
    )
    if not os.path.exists(market_data_info_path):
        raise FileNotFoundError(f"Market data file not found: {market_data_info_path}")

    logger = _configure_run_logger(resolved_symbol)
    run_mode_var = _parse_run_mode(run_mode)
    # create environment
    with open(market_data_info_path, "rb") as f:
        env_data_pkl = pickle.load(f)
    parsed_start = datetime.strptime(start_time, "%Y-%m-%d").date()
    parsed_end = datetime.strptime(end_time, "%Y-%m-%d").date()
    validate_symbol_in_market_data(
        env_data_pkl=env_data_pkl,
        symbol=resolved_symbol,
        start_date=parsed_start,
        end_date=parsed_end,
    )
    environment = MarketEnvironment(
        symbol=resolved_symbol,
        env_data_pkl=env_data_pkl,
        start_date=parsed_start,
        end_date=parsed_end,
    )
    if run_mode_var == RunMode.Train:
        the_agent = LLMAgent.from_config(config)
    else:
        trained_agent_path = resolve_path(
            trained_agent_path,
            "FINMEM_TRAINED_AGENT_PATH",
            "",
        )
        if not trained_agent_path:
            raise ValueError("trained_agent_path is required in test mode")
        the_agent = LLMAgent.load_checkpoint(path=os.path.join(trained_agent_path, "agent_1"))  # type: ignore
    # start simulation
    pbar = tqdm(total=environment.simulation_length)
    while True:
        logger.info(f"Step {the_agent.counter}")
        the_agent.counter += 1
        market_info = environment.step()
        logger.info(f"Date {market_info[0]}")
        logger.info(f"Record {market_info[-2]}")
        if market_info[-1]:  # if done break
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore
        pbar.update(1)
        # save checkpoint every time, openai api is not stable
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)
    # save result after finish
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


@app.command(
    "sim-checkpoint",
    help="Start Simulation from checkpoint",
    rich_help_panel="Simulation",
)
def sim_checkpoint(
    checkpoint_path: Union[str, None] = typer.Option(
        None,
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: Union[str, None] = typer.Option(
        None,
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    config_path: Union[str, None] = typer.Option(
        None,
        "-cp",
        "--config-path",
        help="config file path",
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
    trading_symbol: Union[str, None] = typer.Option(
        None,
        "-sym",
        "--trading-symbol",
        help="Optional symbol override used only when config does not set general.trading_symbol",
    ),
    market_mode: Union[str, None] = typer.Option(
        None,
        "-mm",
        "--market-mode",
        help="Market mode override (US or VN). Defaults to FINMEM_MARKET_MODE/FINMEM_MARKET when provided.",
    ),
) -> None:
    (
        config_path,
        checkpoint_path,
        result_path,
        _,
        resolved_symbol,
        _,
    ) = _prepare_common_runtime(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        trading_symbol=trading_symbol,
        market_mode=market_mode,
    )

    logger = _configure_run_logger(resolved_symbol)
    run_mode_var = _parse_run_mode(run_mode)
    # load env & agent from checkpoint
    environment = MarketEnvironment.load_checkpoint(
        path=os.path.join(checkpoint_path, "env")
    )
    if getattr(environment, "symbol", None) != resolved_symbol:
        raise ValueError(
            f"Checkpoint symbol '{environment.symbol}' does not match resolved trading_symbol '{resolved_symbol}'."
        )
    the_agent = LLMAgent.load_checkpoint(path=os.path.join(checkpoint_path, "agent_1"))
    pbar = tqdm(total=environment.simulation_length)
    # run simulation
    while True:
        logger.info(f"Step {the_agent.counter}")
        the_agent.counter += 1
        market_info = environment.step()
        if market_info[-1]:
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore
        pbar.update(1)
        # save checkpoint every time, openai api is not stable
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)
    # save result after finish
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


@app.command("sim-rl", help="Train PPO/DQN/A2C agents and plot", rich_help_panel="Simulation")
def sim_rl(
    algorithm: str = typer.Option(
        "all",
        "-alg",
        "--algorithm",
        help="RL algorithm: ppo, dqn, a2c, or all",
    ),
    market_data_info_path: Union[str, None] = typer.Option(
        None,
        "-mdp",
        "--market-data-path",
        help="The environment data pickle path",
    ),
    train_start: str = typer.Option(
        "2021-08-17",
        "--train-start",
        help="Training start date",
    ),
    train_end: str = typer.Option(
        "2022-10-05",
        "--train-end",
        help="Training end date",
    ),
    test_start: str = typer.Option(
        "2022-10-06",
        "--test-start",
        help="Test start date",
    ),
    test_end: str = typer.Option(
        "2023-04-10",
        "--test-end",
        help="Test end date",
    ),
    episodes: int = typer.Option(20, "-ep", "--episodes", help="Training episodes"),
    window: int = typer.Option(10, "-w", "--window", help="Feature window size"),
    transaction_cost: float = typer.Option(
        0.001,
        "-tc",
        "--transaction-cost",
        help="Per-step transaction cost for position changes",
    ),
    trading_symbol: Union[str, None] = typer.Option(
        None,
        "-sym",
        "--trading-symbol",
        help="Optional symbol override used only when config does not set general.trading_symbol",
    ),
    market_mode: Union[str, None] = typer.Option(
        None,
        "-mm",
        "--market-mode",
        help="Market mode override (US or VN). Defaults to FINMEM_MARKET_MODE/FINMEM_MARKET when provided.",
    ),
    finmem_state_dict_path: Union[str, None] = typer.Option(
        None,
        "--finmem-state-dict",
        help="Optional FinMem state_dict.pkl to plot alongside",
    ),
    actions_output_dir: Union[str, None] = typer.Option(
        None,
        "--actions-output-dir",
        help="Directory to save aligned RL action artifacts used by 5-measure metrics/plots.",
    ),
    save_path: Union[str, None] = typer.Option(
        None,
        "--save-path",
        help="Output path for plot image",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible RL training",
    ),
    retry_count: int = typer.Option(
        2,
        "--retry-count",
        help="Number of retry attempts per RL baseline after the first failed attempt.",
    ),
    retry_seed_step: int = typer.Option(
        101,
        "--retry-seed-step",
        help="Seed increment applied on each retry attempt.",
    ),
) -> None:
    if retry_count < 0:
        raise ValueError("retry_count must be >= 0")
    if retry_seed_step <= 0:
        raise ValueError("retry_seed_step must be > 0")

    set_global_seed(seed)
    config_path = resolve_path(None, "FINMEM_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    config = toml.load(config_path)
    resolved_symbol = resolve_trading_symbol(
        config=config,
        cli_symbol=trading_symbol,
        default_symbol=DEFAULT_TRADING_SYMBOL,
    )
    resolved_market_mode = resolve_market_mode(
        config=config,
        cli_market_mode=market_mode,
        default_market_mode=DEFAULT_MARKET_MODE,
    )
    actions_output_dir = _resolve_actions_output_dir(actions_output_dir)

    market_data_info_path = resolve_path(
        market_data_info_path,
        "FINMEM_MARKET_DATA_PATH",
        _default_market_data_path(resolved_symbol),
    )
    if not os.path.exists(market_data_info_path):
        raise FileNotFoundError(f"Market data file not found: {market_data_info_path}")

    env_data = load_env_data(market_data_info_path)
    parsed_train_start = datetime.strptime(train_start, "%Y-%m-%d").date()
    parsed_train_end = datetime.strptime(train_end, "%Y-%m-%d").date()
    parsed_test_start = datetime.strptime(test_start, "%Y-%m-%d").date()
    parsed_test_end = datetime.strptime(test_end, "%Y-%m-%d").date()

    validate_symbol_in_market_data(
        env_data_pkl=env_data,
        symbol=resolved_symbol,
        start_date=parsed_train_start,
        end_date=parsed_train_end,
    )
    validate_symbol_in_market_data(
        env_data_pkl=env_data,
        symbol=resolved_symbol,
        start_date=parsed_test_start,
        end_date=parsed_test_end,
    )

    train_dates, train_prices = extract_price_series(
        env_data=env_data,
        symbol=resolved_symbol,
        start=parsed_train_start,
        end=parsed_train_end,
    )
    test_dates, test_prices = extract_price_series(
        env_data=env_data,
        symbol=resolved_symbol,
        start=parsed_test_start,
        end=parsed_test_end,
    )

    algo = algorithm.lower().strip()
    if algo == "all":
        requested_algorithms = ["DQN", "A2C", "PPO"]
    elif algo in {"dqn", "a2c", "ppo"}:
        requested_algorithms = [algo.upper()]
    else:
        raise ValueError("algorithm must be one of: ppo, dqn, a2c, all")

    algo_returns_by_label = {}
    action_artifacts = {}
    for index, algo_label in enumerate(requested_algorithms):
        base_seed = int(seed) + index * 1000
        aligned_positions, algo_returns, used_seed, used_attempt = _run_rl_baseline_with_retry(
            algorithm=algo_label,
            train_dates=train_dates,
            train_prices=train_prices,
            test_dates=test_dates,
            test_prices=test_prices,
            window=window,
            transaction_cost=transaction_cost,
            episodes=episodes,
            seed=base_seed,
            retry_count=retry_count,
            retry_seed_step=retry_seed_step,
        )
        action_path = _save_rl_actions(
            output_dir=actions_output_dir,
            symbol=resolved_symbol,
            algorithm=algo_label,
            market_mode=resolved_market_mode,
            start=test_start,
            end=test_end,
            actions=aligned_positions,
            seed=used_seed,
            attempt=used_attempt,
        )
        typer.secho(
            f"Saved {algo_label} actions: {action_path} (seed={used_seed}, attempt={used_attempt})",
            fg=typer.colors.GREEN,
        )
        algo_returns_by_label[algo_label] = algo_returns
        action_artifacts[algo_label] = action_path

    if algo == "all":
        missing = [label for label in ["DQN", "A2C", "PPO"] if label not in action_artifacts]
        if missing:
            raise ValueError(
                "Missing RL action artifacts after retries for: "
                + ", ".join(missing)
            )

    labels = []
    returns = []

    buy_hold_actions = [1] * max(0, len(test_prices) - 1)
    returns.insert(0, actions_to_cum_returns(test_prices.tolist(), buy_hold_actions))
    labels.insert(0, "Buy & Hold")

    if finmem_state_dict_path:
        finmem_actions = load_finmem_actions_aligned(
            state_dict_path=finmem_state_dict_path,
            start=test_start,
            end=test_end,
            date_series=test_dates,
        )
        finmem_returns = actions_to_cum_returns(test_prices.tolist(), finmem_actions)
        returns.insert(1, finmem_returns)
        labels.insert(1, "FinMem")

    for algo_label in requested_algorithms:
        returns.append(algo_returns_by_label[algo_label])
        labels.append(algo_label)

    suffix = algo if algo != "all" else "ppo_dqn_a2c"
    output_path = save_path or os.path.join(
        "figures",
        f"{resolved_symbol}_{suffix}_comparison.png",
    )

    plot_cumulative_returns(
        dates=test_dates,
        return_lists=returns,
        labels=labels,
        ticker=resolved_symbol,
        file_path=output_path,
    )

    # Reference values are approximated from the pasted paper figure for quick gap checks.
    paper_ref = {
        "DQN": 0.34,
        "PPO": 0.02,
        "A2C": 0.16,
    }
    final_by_label = {
        label: float(series[-1]) if len(series) > 0 else 0.0
        for label, series in zip(labels, returns)
    }
    print("\nFinal cumulative return summary:")
    for label in labels:
        print(f"  {label}: {final_by_label[label]:.4f}")

    print("\nDifference vs paper figure (approx):")
    for label in ["DQN", "PPO", "A2C"]:
        if label in final_by_label:
            diff = final_by_label[label] - paper_ref[label]
            print(f"  {label}: {final_by_label[label]:.4f} vs {paper_ref[label]:.4f} (delta {diff:+.4f})")


if __name__ == "__main__":
    app()
