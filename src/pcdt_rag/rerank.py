"""Cross-encoder rerankers, run on ONNX Runtime like the embedder.

A cross-encoder reads the question and the chunk together and outputs one relevance
score, so it is far slower than comparing two precomputed vectors. It only reorders the
few candidates that the first-stage retrieval returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_TOKENS = 512


@dataclass(frozen=True)
class Model:
    repo: str
    file: str
    # The int8 export quantizes activations per batch, so a pair's score depends on the
    # padding of the others in its batch. One pair per run keeps the scores comparable.
    batch: int = 8


MODELS = {
    # 118M parameters, distilled from mMARCO (MS MARCO translated into 14 languages).
    "minilm": Model("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "onnx/model.onnx"),
    # 568M parameters. The fp32 export needs 2.2 GB of RAM, so this uses the int8 one.
    "bge": Model("onnx-community/bge-reranker-v2-m3-ONNX", "onnx/model_int8.onnx", batch=1),
}


class CrossEncoder:
    def __init__(self, name: str) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model = MODELS[name]
        self.batch = model.batch
        self.tokenizer = Tokenizer.from_pretrained(model.repo)
        # Chunks have 500 tokens, so with the question and 4 special tokens most pairs pass
        # 512 and lose the last few tokens of the chunk (about 13). The 50-token overlap puts
        # that text at the start of the next chunk. The MiniLM model accepts no more than 512.
        self.tokenizer.enable_truncation(MAX_TOKENS, strategy="only_second")
        self.tokenizer.enable_padding()
        path = hf_hub_download(model.repo, model.file)
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def scores(self, query: str, passages: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(passages), self.batch):
            pairs = [(query, p) for p in passages[i : i + self.batch]]
            encodings = self.tokenizer.encode_batch(pairs)
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            feed = {
                "input_ids": ids,
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
            out.append(self.session.run(None, feed)[0][:, 0])
        return np.concatenate(out) if out else np.zeros(0)
