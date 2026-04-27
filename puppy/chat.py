import os
import httpx
import json
import time
import subprocess
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

### when use tgi model
api_key = '-' 

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt


class LongerThanContextError(Exception):
    pass


def _extract_json_candidate(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return text

class ChatOpenAICompatible(ABC):
    def __init__(
        self,
        end_point: str,
        model="gemini-pro",
        system_message: str = "You are a helpful assistant.",
        other_parameters: Union[Dict[str, Any], None] = None,
    ):
        api_key = os.environ.get("CEREBRAS_API_KEY", "-")
        self.end_point = end_point
        self.model = model
        self.system_message = system_message
        self.other_parameters = (
            {} if other_parameters is None else dict(other_parameters)
        )
        self.openai_compatible = bool(
            self.other_parameters.get("openai_compatible", False)
        )
        cfg_raw = self.other_parameters.get("api_key")
        cfg_key = str(cfg_raw) if cfg_raw is not None else None

        def _placeholder(k: Optional[str]) -> bool:
            if k is None:
                return True
            s = k.strip()
            return (
                not s
                or s in ("-", "EMPTY")
                or "enter your" in s.lower()
            )

        if self.openai_compatible:
            self.api_key = (
                os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or (cfg_key if not _placeholder(cfg_key) else "")
                or os.environ.get("CEREBRAS_API_KEY", "-")
            )
        else:
            self.api_key = (
                cfg_key if not _placeholder(cfg_key) else api_key
            )
        if self.openai_compatible and self.end_point.rstrip("/").endswith("/v1"):
            self.end_point = f"{self.end_point.rstrip('/')}/chat/completions"

        if self.model.startswith("gemini-pro"):
            proc_result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
            access_token = proc_result.stdout.strip()
            self.headers = {     "Authorization": f"Bearer {access_token}",
                                "Content-Type": "application/json",
                            }
        elif self.model.startswith("tgi"):
            self.headers = {
                        'Content-Type': 'application/json'
                    }   
        else:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        if self.openai_compatible and "openrouter.ai" in self.end_point.lower():
            self.headers["HTTP-Referer"] = os.environ.get(
                "OPENROUTER_HTTP_REFERER", "http://localhost"
            )
            self.headers["X-Title"] = os.environ.get("OPENROUTER_APP_NAME", "FinMem")

    def parse_response(self, response: httpx.Response) -> str:
        if self.openai_compatible or self.model.startswith("gpt"):
            response_out = response.json()
            content = response_out["choices"][0]["message"].get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return _extract_json_candidate(str(content))
        elif self.model.startswith("gemini-pro"):
            response_out = response.json()
            return response_out["candidates"][0]["content"]["parts"][0]["text"]
        elif self.model.startswith("tgi"):
            response_out = response.json()#[0]
            return response_out["generated_text"]
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

    @staticmethod
    def _is_rate_limit_error(err: Exception) -> bool:
        err_text = str(err).lower()
        return (
            "429" in err_text
            or "rate limit" in err_text
            or "too_many_tokens_error" in err_text
            or "token_quota_exceeded" in err_text
        )

    def guardrail_endpoint(self) -> Callable:
        json_system_suffix = (
            f"{self.system_message}\nYou are only capable of communicating with valid JSON, "
            "and no other text."
        )

        def end_point(
            *args: Any,
            prompt: Optional[str] = None,
            input: Optional[str] = None,
            messages: Optional[List[Dict[str, str]]] = None,
            _instructions: Optional[str] = None,
            **kwargs: Any,
        ) -> str:
            # guardrails 0.6+ calls custom LLMs with messages=... (Runner.call).
            if messages:
                input_str = [dict(m) for m in messages]
                if not any(m.get("role") == "system" for m in input_str):
                    input_str.insert(
                        0,
                        {"role": "system", "content": json_system_suffix},
                    )
            else:
                user_content: Optional[str] = prompt if prompt is not None else input
                if user_content is None and args:
                    user_content = str(args[0])
                if user_content is None:
                    raise ValueError(
                        "LLM guard endpoint needs messages=, prompt=, input=, or a positional prompt."
                    )
                input_str = [
                    {"role": "system", "content": json_system_suffix},
                    {"role": "user", "content": user_content},
                ]
            
            # For custom OpenAI-compatible endpoints, use the configured model only.
            # For default Cerebras flow, keep fallback models for resilience.
            if self.openai_compatible:
                models_to_try = [self.model]
            else:
                fallback_models = ["llama3.1-8b", "gpt-oss-120b", "qwen-3-235b-a22b-instruct-2507", "zai-glm-4.7"]
                models_to_try = [self.model]
                for m in fallback_models:
                    if m != self.model:
                        models_to_try.append(m)

            retry_max_attempts = int(os.environ.get("FINMEM_LLM_RETRY_MAX_ATTEMPTS", "5"))
            retry_base_wait_seconds = int(os.environ.get("FINMEM_LLM_RETRY_BASE_WAIT_SECONDS", "8"))
            retry_max_wait_seconds = int(os.environ.get("FINMEM_LLM_RETRY_MAX_WAIT_SECONDS", "120"))
            
            last_error = None
            for model_name in models_to_try:
                attempt = 0
                while attempt < retry_max_attempts:
                    try:
                        if model_name.startswith("gemini-pro"):
                            input_prompts = {"role": "USER",
                                            "parts": { "text": input_str[1]["content"]}
                                                }
                            payload = {"contents": input_prompts,
                                        "generation_config": {
                                                            "temperature": 0.2,
                                                            "top_p": 0.1,
                                                            "top_k": 16,
                                                            "max_output_tokens": 2048,
                                                            "candidate_count": 1,
                                                            "stop_sequences": []
                                                            },
                                        "safety_settings": {
                                                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                                            "threshold": "BLOCK_LOW_AND_ABOVE"
                                                            }
                                    }
                            response = httpx.post(url = self.end_point, headers= self.headers, json=payload, timeout=600.0 )
                            response.raise_for_status()
                            return self.parse_response(response)

                        elif model_name.startswith("tgi"):
                            llama_input_str = build_llama2_prompt(input_str)
                            payload = {
                                "inputs": llama_input_str,
                                "parameters": {
                                    "do_sample": True,
                                    "top_p": 0.6,
                                    "temperature": 0.8,
                                    "top_k": 50,
                                    "max_new_tokens": 256,
                                    "repetition_penalty": 1.03,
                                    "stop": ["</s>"]
                                }
                            }
                            response = httpx.post(
                                self.end_point, headers=self.headers, json=payload, timeout=600.0
                            )
                            response.raise_for_status()
                            return self.parse_response(response)
                        elif self.openai_compatible:
                            payload = {
                                "model": model_name,
                                "messages": input_str,
                                "temperature": self.other_parameters.get("temperature", 0.2),
                            }
                            if self.other_parameters.get("json_response_format", True):
                                payload["response_format"] = {"type": "json_object"}
                            if "max_tokens" in self.other_parameters:
                                payload["max_tokens"] = self.other_parameters["max_tokens"]
                            if "top_p" in self.other_parameters:
                                payload["top_p"] = self.other_parameters["top_p"]
                            response = httpx.post(
                                self.end_point,
                                headers=self.headers,
                                json=payload,
                                timeout=600.0,
                            )
                            response.raise_for_status()
                            return self.parse_response(response)
                        else:
                            from langchain_cerebras import ChatCerebras
                            from langchain_core.messages import SystemMessage, HumanMessage
                            cerebras_api_key = os.environ.get("CEREBRAS_API_KEY", "-")

                            # Use model_name for the attempt
                            chat = ChatCerebras(model=model_name, api_key=cerebras_api_key)
                            msgs = [
                                SystemMessage(content=input_str[0]["content"]),
                                HumanMessage(content=input_str[1]["content"])
                            ]
                            res = chat.invoke(msgs)
                            return res.content

                    except LongerThanContextError:
                        # Don't retry on context length errors as it's likely a persistent issue for this input
                        raise
                    except Exception as e:
                        last_error = e
                        attempt += 1
                        if self._is_rate_limit_error(e) and attempt < retry_max_attempts:
                            wait_seconds = min(
                                retry_base_wait_seconds * (2 ** (attempt - 1)),
                                retry_max_wait_seconds,
                            )
                            time.sleep(wait_seconds)
                            continue
                        break
            
            # If all models failed
            if last_error:
                raise last_error
            return "All models failed to provide a response."

        return end_point

