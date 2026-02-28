import os
import numpy as np
from typing import List, Union
from langchain_huggingface import HuggingFaceEndpointEmbeddings


class HuggingFaceEmb:
    """
    Embedding function using HuggingFace Inference API (router.huggingface.co).
    Uses langchain_huggingface.HuggingFaceEndpointEmbeddings which targets
    the current HF Inference API router endpoint.
    """

    def __init__(
        self,
        hf_token: Union[str, None] = None,
        embedding_model: str = "intfloat/multilingual-e5-large",
        chunk_size: int = 5000,
        verbose: bool = False,
        **kwargs
    ) -> None:
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.emb_model = HuggingFaceEndpointEmbeddings(
            model=embedding_model,
            huggingfacehub_api_token=self.hf_token,
        )
        self.model_name = embedding_model

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        return self.emb_model.embed_documents(texts=text)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        return np.array(self._emb(text)).astype("float32")

    def get_embedding_dimension(self):
        match self.model_name:
            case "intfloat/multilingual-e5-large":
                return 1024
            case "sentence-transformers/all-MiniLM-L6-v2":
                return 384
            case "BAAI/bge-m3":
                return 1024
            case _:
                return len(self._emb(["test"])[0])
