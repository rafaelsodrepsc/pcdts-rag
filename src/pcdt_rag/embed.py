"""Embed the chunks with multilingual-e5 (data/processed/embeddings.npz).

The model runs on ONNX Runtime, so the project does not need PyTorch. The ONNX export
is the one published in the model repository.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pcdt_rag.chunk import MODEL
from pcdt_rag.chunk import OUT_FILE as CHUNKS_FILE

OUT_FILE = Path("data/processed/embeddings.npz")
MAX_TOKENS = 512
BATCH = 8


class Embedder:
    def __init__(self, progress: bool = False) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.progress = progress
        self.tokenizer = Tokenizer.from_pretrained(MODEL)
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()
        path = hf_hub_download(MODEL, "onnx/model.onnx")
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH):
            encodings = self.tokenizer.encode_batch(texts[i : i + BATCH])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]
            out.append(mean_pool(hidden, mask))
            if self.progress:
                done = min(i + BATCH, len(texts))
                print(f"\r{done}/{len(texts)}", end="" if done < len(texts) else "\n", flush=True)
        return np.concatenate(out)

    def passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts])


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average the token vectors, ignoring padding, and normalize to unit length."""
    weights = mask[..., None].astype(hidden.dtype)
    pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
    return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    embedder = Embedder(progress=True)
    start = time.perf_counter()
    vectors = embedder.passages([c["text"] for c in chunks])
    elapsed = time.perf_counter() - start
    np.savez(OUT_FILE, ids=np.array([c["id"] for c in chunks]), vectors=vectors)
    print(f"{len(chunks)} chunks in {elapsed:.0f} s, dim {vectors.shape[1]}. Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
