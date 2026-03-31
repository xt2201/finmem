import os
import toml
import typer
import logging
import pickle
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from typing import Union
from puppy import MarketEnvironment, LLMAgent, RunMode
from puppy.runtime_config import (
    DEFAULT_TRADING_SYMBOL,
    expand_symbol_template,
    resolve_path,
    resolve_trading_symbol,
    validate_symbol_in_market_data,
)


# set up
load_dotenv()
app = typer.Typer(name="puppy")
warnings.filterwarnings("ignore")

DEFAULT_CONFIG_PATH = os.path.join("config", "finmem_cerebras_config.toml")


def _default_market_data_path(symbol: str) -> str:
    return os.path.join("data", "03_model_input", f"{symbol.lower()}.pkl")


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
) -> None:
    config_path = resolve_path(config_path, "FINMEM_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    checkpoint_path = resolve_path(
        checkpoint_path,
        "FINMEM_CHECKPOINT_PATH",
        os.path.join("data", "06_train_checkpoint"),
    )
    result_path = resolve_path(
        result_path,
        "FINMEM_RESULT_PATH",
        os.path.join("data", "05_train_model_output"),
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # load config
    config = toml.load(config_path)
    resolved_symbol = resolve_trading_symbol(
        config=config,
        cli_symbol=trading_symbol,
        default_symbol=DEFAULT_TRADING_SYMBOL,
    )
    config.setdefault("general", {})["trading_symbol"] = resolved_symbol
    if "character_string" in config["general"]:
        config["general"]["character_string"] = expand_symbol_template(
            config["general"]["character_string"],
            resolved_symbol,
        )

    market_data_info_path = resolve_path(
        market_data_info_path,
        "FINMEM_MARKET_DATA_PATH",
        _default_market_data_path(resolved_symbol),
    )
    if not os.path.exists(market_data_info_path):
        raise FileNotFoundError(f"Market data file not found: {market_data_info_path}")

    # set up logging
    os.makedirs(os.path.join("data", "04_model_output_log"), exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(
            "data",
            "04_model_output_log",
            f"{resolved_symbol}_run.log",
        ),
        mode="a",
    )
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)
    # verify run mode
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("Run mode must be train or test")
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
) -> None:
    config_path = resolve_path(config_path, "FINMEM_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    checkpoint_path = resolve_path(
        checkpoint_path,
        "FINMEM_CHECKPOINT_PATH",
        os.path.join("data", "06_train_checkpoint"),
    )
    result_path = resolve_path(
        result_path,
        "FINMEM_RESULT_PATH",
        os.path.join("data", "05_train_model_output"),
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # load config
    config = toml.load(config_path)
    resolved_symbol = resolve_trading_symbol(
        config=config,
        cli_symbol=trading_symbol,
        default_symbol=DEFAULT_TRADING_SYMBOL,
    )
    config.setdefault("general", {})["trading_symbol"] = resolved_symbol
    if "character_string" in config["general"]:
        config["general"]["character_string"] = expand_symbol_template(
            config["general"]["character_string"],
            resolved_symbol,
        )

    # set up logging
    os.makedirs(os.path.join("data", "04_model_output_log"), exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(
            "data",
            "04_model_output_log",
            f"{resolved_symbol}_run.log",
        ),
        mode="a",
    )
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)
    # verify run mode
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("Run mode must be train or test")
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


if __name__ == "__main__":
    app()
