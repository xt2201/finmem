import argparse
import os

from puppy import LLMAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export action dataframe from checkpoint")
    parser.add_argument(
        "--agent-checkpoint",
        default=os.environ.get("FINMEM_SAVE_AGENT_CHECKPOINT", "./data/08_test_checkpoint/agent_1"),
        help="Path to agent checkpoint directory",
    )
    parser.add_argument(
        "--ticker",
        default=os.environ.get("FINMEM_TRADING_SYMBOL", "TSLA"),
        help="Ticker used for output file naming when --output is not provided",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <ticker>_actions.csv",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    output_path = args.output or f"{ticker.lower()}_actions.csv"
    agent = LLMAgent.load_checkpoint(args.agent_checkpoint)
    df = agent.portfolio.get_action_df()
    df.write_csv(output_path)
    print(f"Saved actions to: {output_path}")