"""Retrieval pipelines over the chunk index, and a command line to try them.

uv run pcdt-search "qual a dose máxima de levodopa?" [--pipeline hibrida+minilm]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import textwrap
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from pcdt_rag.chunk import OUT_FILE as CHUNKS_FILE
from pcdt_rag.chunk import read_pages
from pcdt_rag.embed import OUT_FILE as EMBEDDINGS_FILE
from pcdt_rag.embed import Embedder
from pcdt_rag.extract import OUT_FILE as PAGES_FILE
from pcdt_rag.lexical import BM25, tokenize
from pcdt_rag.methodology import methodology_pages
from pcdt_rag.rerank import MODELS, CrossEncoder

SNIPPET = 160
# Results taken from each retriever before fusing, and candidates given to the reranker.
CANDIDATES = 50
RERANK_DEPTH = 20
# Constant of reciprocal rank fusion, from the original paper (Cormack et al., 2009).
RRF_K = 60
FIRST_STAGES = ("vetorial", "bm25", "hibrida")
PIPELINES = (*FIRST_STAGES, *(f"hibrida+{m}" for m in MODELS))
DEFAULT_PIPELINE = "hibrida+minilm"


def load_chunks() -> list[dict]:
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_index(chunks: list[dict]) -> np.ndarray:
    data = np.load(EMBEDDINGS_FILE)
    if list(data["ids"]) != [c["id"] for c in chunks]:
        raise SystemExit(f"{EMBEDDINGS_FILE} does not match {CHUNKS_FILE}: run pcdt-embed again")
    return data["vectors"]


def search(queries: np.ndarray, docs: np.ndarray, allowed: np.ndarray, top: int) -> np.ndarray:
    """Indices of the `top` most similar allowed docs for each query (vectors are unit length)."""
    scores = queries @ docs.T
    scores[:, ~allowed] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :top]


def top_k(scores: np.ndarray, allowed: np.ndarray, top: int) -> list[int]:
    """Indices of the `top` highest positive scores among the allowed docs."""
    scores = np.where(allowed & (scores > 0), scores, -np.inf)
    return [int(i) for i in np.argsort(-scores, kind="stable")[:top] if scores[i] > -np.inf]


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to the docs it ranks.

    It uses only the positions, so the cosine similarity and the BM25 score, which are on
    unrelated scales, never need to be normalized against each other.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] += 1 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])


class Retriever:
    def __init__(self, chunks: list[dict], allowed: np.ndarray | None = None) -> None:
        self.chunks = chunks
        self.vectors = load_index(chunks)
        self.allowed = np.ones(len(chunks), dtype=bool) if allowed is None else allowed
        self.bm25 = BM25([c["text"] for c in chunks])
        self.embedder = Embedder()
        self.rerankers: dict[str, CrossEncoder] = {}

    def dense(self, query: str, top: int) -> list[int]:
        vector = self.embedder.queries([query])
        return search(vector, self.vectors, self.allowed, top)[0].tolist()

    def lexical(self, query: str, top: int) -> list[int]:
        return top_k(self.bm25.scores(query), self.allowed, top)

    def hybrid(self, query: str, top: int) -> list[int]:
        return rrf([self.dense(query, CANDIDATES), self.lexical(query, CANDIDATES)])[:top]

    def rerank(self, model: str, query: str, candidates: list[int], top: int) -> list[int]:
        if model not in self.rerankers:
            self.rerankers[model] = CrossEncoder(model)
        scores = self.rerankers[model].scores(query, [self.chunks[i]["text"] for i in candidates])
        return [candidates[i] for i in np.argsort(-scores, kind="stable")[:top]]

    def run(self, pipeline: str, query: str, top: int) -> list[int]:
        """Chunk indices for `pipeline`, a first stage optionally followed by "+reranker"."""
        stage, _, model = pipeline.partition("+")
        first = {"vetorial": self.dense, "bm25": self.lexical, "hibrida": self.hybrid}[stage]
        if not model:
            return first(query, top)
        return self.rerank(model, query, first(query, max(top, RERANK_DEPTH)), top)


def snippet(text: str, query: str, bm25: BM25, size: int = SNIPPET) -> str:
    """`size` characters of the chunk, starting a little before the rarest query term in it."""
    flat = " ".join(text.split())
    # One base letter per character, so an offset in `folded` is the same offset in `flat`.
    folded = "".join(unicodedata.normalize("NFKD", ch.lower())[0] for ch in flat)
    terms = sorted(
        (t for t in set(tokenize(query)) if t in bm25.weights),
        key=lambda t: len(bm25.weights[t][0]),
    )
    start = 0
    for term in terms:
        if match := re.search(rf"\b{term}\b", folded):
            start = max(0, match.start() - size // 4)
            break
    return ("..." if start else "") + flat[start : start + size].strip() + "..."


def corpus_methodology() -> set[tuple[str, int]]:
    """(slug, page) of every page inside a methodology appendix."""
    return {
        (pages[0].slug, n) for pages in read_pages(PAGES_FILE) for n in methodology_pages(pages)
    }


def outside_methodology(chunks: list[dict], methodology: set[tuple[str, int]]) -> np.ndarray:
    """Mask of the chunks that are not entirely inside a methodology appendix."""
    return np.array(
        [
            not (
                (c["slug"], c["page_start"]) in methodology
                and (c["slug"], c["page_end"]) in methodology
            )
            for c in chunks
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--pipeline", choices=PIPELINES, default=DEFAULT_PIPELINE)
    parser.add_argument("--sem-metodologia", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks()
    allowed = None
    if args.sem_metodologia:
        allowed = outside_methodology(chunks, corpus_methodology())
    retriever = Retriever(chunks, allowed)
    width = min(shutil.get_terminal_size().columns, 100)
    color = sys.stdout.isatty()
    for rank, i in enumerate(retriever.run(args.pipeline, args.question, args.k), start=1):
        c = chunks[i]
        pages = f"p. {c['page_start']}" + (
            f"-{c['page_end']}" if c["page_end"] != c["page_start"] else ""
        )
        header = f"{rank}. {c['slug']}, {pages}"
        if color:
            header = f"\033[1;33m{header}\033[0m"
        text = textwrap.fill(
            snippet(c["text"], args.question, retriever.bm25),
            width=width,
            initial_indent="   ",
            subsequent_indent="   ",
        )
        print(f"{header}\n{text}\n")


if __name__ == "__main__":
    main()
