import os
import httpx
import json
import subprocess
from abc import ABC
from typing import Callable, Union, Dict, Any, Union

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
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            self.other_parameters = {} if other_parameters is None else other_parameters

    def parse_response(self, response: httpx.Response) -> str:
        if self.model.startswith("gpt"):
            response_out = response.json()
            return response_out["choices"][0]["message"]["content"]
        elif self.model.startswith("gemini-pro"):
            response_out = response.json()
            return response_out["candidates"][0]["content"]["parts"][0]["text"]
        elif self.model.startswith("tgi"):
            response_out = response.json()#[0]
            return response_out["generated_text"]
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

    def guardrail_endpoint(self) -> Callable:
        def end_point(input: str, **kwargs) -> str:
            input_str = [
                {"role": "system", "content": "You are a helpful assistant only capable of communicating with valid JSON, and no other text."},
                {"role": "user", "content": f"{input}"},
            ]
            
            # Models to try in order: current model, then fallbacks
            fallback_models = ["gpt-oss-120b", "zai-glm-4.7"]
            models_to_try = [self.model]
            for m in fallback_models:
                if m != self.model:
                    models_to_try.append(m)
            
            last_error = None
            for model_name in models_to_try:
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
                    # Continue to next fallback model
                    continue
            
            # If all models failed
            if last_error:
                raise last_error
            return "All models failed to provide a response."

        return end_point

