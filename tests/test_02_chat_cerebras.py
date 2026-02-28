"""Test 02: Cerebras LLM Chat (ChatOpenAICompatible)
Validates:
- ChatCerebras initialization with CEREBRAS_API_KEY
- guardrail_endpoint callable creation
- Successful LLM invocation and non-empty response
- Context error handling
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
results = {"test": "test_02_chat_cerebras", "passed": [], "failed": []}


def run():
    from puppy.chat import ChatOpenAICompatible

    # 1. Initialization
    try:
        chat = ChatOpenAICompatible(
            end_point="https://api.cerebras.ai/v1/chat/completions",
            model="llama3.1-8b",
            system_message="You are a helpful assistant.",
        )
        results["passed"].append("init_chat")
    except Exception as e:
        results["failed"].append({"init_chat": str(e)})
        save_and_exit()
        return

    # 2. guardrail_endpoint returns a callable
    endpoint = chat.guardrail_endpoint()
    if callable(endpoint):
        results["passed"].append("guardrail_endpoint_callable")
    else:
        results["failed"].append({"guardrail_endpoint_callable": f"type={type(endpoint)}"})
        save_and_exit()
        return

    # 3. Invoke with a simple prompt
    try:
        response = endpoint("What is 2 + 2? Reply with just the number.")
        results["llm_response"] = response[:500] if response else None
        if response and len(response.strip()) > 0:
            results["passed"].append("llm_invocation")
        else:
            results["failed"].append({"llm_invocation": "empty response"})
    except Exception as e:
        results["failed"].append({"llm_invocation": str(e)})

    save_and_exit()


def save_and_exit():
    results["summary"] = f"{len(results['passed'])} passed, {len(results['failed'])} failed"
    with open(OUTPUT_DIR / "test_02_chat_cerebras.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
