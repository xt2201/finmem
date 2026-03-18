from pathlib import Path
import importlib.util

try:
    from model_wrapper import Model_Factory
except ModuleNotFoundError:
    # Fallback for repositories where wrapper is still named 03-model_wrapper.py.
    wrapper_path = Path(__file__).with_name("03-model_wrapper.py")
    if not wrapper_path.exists():
        raise
    spec = importlib.util.spec_from_file_location("model_wrapper", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model wrapper from {wrapper_path}")
    spec.loader.exec_module(module)
    Model_Factory = module.Model_Factory

import os
import pandas as pd
import concurrent.futures
import threading


def _load_summary_model():
    provider = os.environ.get("FINMEM_SUMMARY_PROVIDER", "cerebras").lower()
    model_name = os.environ.get("FINMEM_SUMMARY_MODEL", "llama3.1-8b")

    if provider == "cerebras":
        api_key = os.environ.get("CEREBRAS_API_KEY")
    elif provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY")
    elif provider == "dummy":
        api_key = None
    else:
        raise ValueError(
            "FINMEM_SUMMARY_PROVIDER must be one of: cerebras, together, dummy"
        )

    return Model_Factory.create_model(provider, key=api_key, model_name=model_name)


model = _load_summary_model()
summary = model.summarize


# Paths can be overridden via environment variables for local runs.
SOURCE_PATH = os.environ.get(
    "FINMEM_SUMMARY_SOURCE_PATH", "/home/hfsladmin/Projects/finmem/Cleaaned_Data"
)
DEST_PATH = os.environ.get(
    "FINMEM_SUMMARY_DEST_PATH", "/home/hfsladmin/Projects/finmem/add_summary_data"
)
TEMP_PATH = os.environ.get(
    "FINMEM_SUMMARY_TEMP_PATH", "/home/hfsladmin/Projects/finmem/add_summary_data_tmp"
)


os.makedirs(DEST_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)


def process_row(row, lock, df, df_name):
    # Summarize the text body for a single news row.
    result = summary(row["body"])

    with lock:
        df.at[row.name, "summary"] = result
        df.to_csv(os.path.join(TEMP_PATH, df_name), index=False)


def parallel_summary(df, df_name):
    lock = threading.Lock()
    df_copy = df.copy()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_row, row, lock, df_copy, df_name)
            for _, row in df.iterrows()
        ]
        concurrent.futures.wait(futures)
    return df_copy


def process_main(file):
    if file.endswith(".csv"):
        print(f"Processing: {file}")
        df = pd.read_csv(os.path.join(SOURCE_PATH, file))
        df["summary"] = None
        ret = parallel_summary(df, file)
        ret.to_csv(os.path.join(DEST_PATH, file), index=False)
        print(f"New DF dumped to {os.path.join(DEST_PATH, file)}")


if __name__ == "__main__":
    file_ls = [f for f in os.listdir(SOURCE_PATH) if f.endswith(".csv")]
    if not file_ls:
        print(f"No CSV files found in {SOURCE_PATH}")
    for file in file_ls:
        process_main(file)
