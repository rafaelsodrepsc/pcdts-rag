"""Search the chunk index from the command line.

uv run pcdt-search "qual a dose máxima de levodopa?" [--sem-metodologia]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap

import numpy as np

from pcdt_rag.chunk import OUT_FILE as CHUNKS_FILE
from pcdt_rag.chunk import read_pages
from pcdt_rag.embed import OUT_FILE as EMBEDDINGS_FILE
from pcdt_rag.embed import Embedder
from pcdt_rag.extract import OUT_FILE as PAGES_FILE
from pcdt_rag.methodology import methodology_pages

SNIPPET = 160


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
    parser.add_argument("--sem-metodologia", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks()
    docs = load_index(chunks)
    allowed = np.ones(len(chunks), dtype=bool)
    if args.sem_metodologia:
        allowed = outside_methodology(chunks, corpus_methodology())
    query = Embedder().queries([args.question])
    width = min(shutil.get_terminal_size().columns, 100)
    color = sys.stdout.isatty()
    for rank, i in enumerate(search(query, docs, allowed, args.k)[0], start=1):
        c = chunks[i]
        pages = f"p. {c['page_start']}" + (
            f"-{c['page_end']}" if c["page_end"] != c["page_start"] else ""
        )
        header = f"{rank}. {c['slug']}, {pages}"
        if color:
            header = f"\033[1;33m{header}\033[0m"
        text = textwrap.fill(
            " ".join(c["text"].split())[:SNIPPET] + "...",
            width=width,
            initial_indent="   ",
            subsequent_indent="   ",
        )
        print(f"{header}  ({float(query[0] @ docs[i]):.3f})\n{text}\n")


if __name__ == "__main__":
    main()
