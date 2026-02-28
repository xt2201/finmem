"""Test 01: HuggingFace Embedding (intfloat/multilingual-e5-large)
Validates:
- Embedding initialization with HF_TOKEN
- Embedding dimension correctness (1024)
- Output shape and dtype
- Cosine similarity between semantically similar/different texts
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
results = {"test": "test_01_embedding", "passed": [], "failed": []}


def run():
    from puppy.embedding import HuggingFaceEmb

    # 1. Initialization
    try:
        emb = HuggingFaceEmb(embedding_model="intfloat/multilingual-e5-large")
        results["passed"].append("init_embedding")
    except Exception as e:
        results["failed"].append({"init_embedding": str(e)})
        save_and_exit()
        return

    # 2. Dimension check
    dim = emb.get_embedding_dimension()
    if dim == 1024:
        results["passed"].append("dimension_check_1024")
    else:
        results["failed"].append({"dimension_check": f"expected 1024, got {dim}"})

    # 3. Single text embedding shape and dtype
    vec = emb("Tesla stock price is rising")
    if vec.shape == (1, 1024) and vec.dtype == np.float32:
        results["passed"].append("single_text_shape_dtype")
    else:
        results["failed"].append({"single_text_shape_dtype": f"shape={vec.shape}, dtype={vec.dtype}"})

    # 4. Batch embedding
    texts = ["Tesla earnings beat expectations", "Apple released a new iPhone", "It rained today"]
    vecs = emb(texts)
    if vecs.shape == (3, 1024):
        results["passed"].append("batch_embedding_shape")
    else:
        results["failed"].append({"batch_embedding_shape": f"expected (3, 1024), got {vecs.shape}"})

    # 5. Cosine similarity: similar texts should score higher than dissimilar
    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_similar = cos_sim(vecs[0], vecs[1])  # both about companies
    sim_dissimilar = cos_sim(vecs[0], vecs[2])  # finance vs weather
    results["cosine_similarity"] = {"similar_pair": sim_similar, "dissimilar_pair": sim_dissimilar}
    if sim_similar > sim_dissimilar:
        results["passed"].append("cosine_similarity_ordering")
    else:
        results["failed"].append({"cosine_similarity_ordering": f"similar={sim_similar}, dissimilar={sim_dissimilar}"})

    save_and_exit()


def save_and_exit():
    results["summary"] = f"{len(results['passed'])} passed, {len(results['failed'])} failed"
    with open(OUTPUT_DIR / "test_01_embedding.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    run()
