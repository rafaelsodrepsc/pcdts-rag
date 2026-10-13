"""Measure retrieval against the golden set (eval/golden.jsonl).

Each question is annotated with the pages that answer it. A retrieved chunk is relevant
when it comes from the same document and covers one of those pages with text: a page that
is only an image cannot be retrieved, and the evaluation counts it as a miss.

Two configurations are compared: the full index, and the index without the chunks that
lie inside the methodology appendices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcdt_rag.chunk import MODEL, read_pages
from pcdt_rag.embed import Embedder
from pcdt_rag.extract import OUT_FILE as PAGES_FILE
from pcdt_rag.retrieve import (
    corpus_methodology,
    load_chunks,
    load_index,
    outside_methodology,
    search,
)

GOLDEN_FILE = Path("eval/golden.jsonl")
OUT_FILE = Path("eval/results/baseline.json")
KS = (1, 5, 10)
TOP = max(KS)
RESAMPLES = 10_000
SEED = 0


@dataclass(frozen=True)
class Question:
    id: str
    slug: str
    pages: tuple[int, ...]
    question: str
    answer: str


def load_golden(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [Question(**{**r, "pages": tuple(r["pages"])}) for r in rows]


def is_relevant(chunk: dict, question: Question, text_pages: set[tuple[str, int]]) -> bool:
    return chunk["slug"] == question.slug and any(
        chunk["page_start"] <= p <= chunk["page_end"] and (question.slug, p) in text_pages
        for p in question.pages
    )


def first_hit(ranked: list[bool]) -> int | None:
    """1-based rank of the first relevant result, or None when there is none."""
    return next((i for i, relevant in enumerate(ranked, start=1) if relevant), None)


def hit_at(ranks: list[int | None], k: int) -> np.ndarray:
    return np.array([r is not None and r <= k for r in ranks], dtype=float)


def reciprocal_rank(ranks: list[int | None]) -> np.ndarray:
    return np.array([1 / r if r is not None else 0.0 for r in ranks])


def bootstrap_ci(values: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> list[float]:
    """95% percentile interval of the mean, resampling the questions."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values))).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metrics(ranks: list[int | None]) -> dict[str, np.ndarray]:
    out = {f"hit@{k}": hit_at(ranks, k) for k in KS}
    out[f"mrr@{TOP}"] = reciprocal_rank(ranks)
    return out


def summarize(values: dict[str, np.ndarray]) -> dict[str, dict]:
    return {name: {"mean": float(v.mean()), "ci95": bootstrap_ci(v)} for name, v in values.items()}


def main() -> None:
    questions = load_golden(GOLDEN_FILE)
    chunks = load_chunks()
    docs = load_index(chunks)
    text_pages = {(p.slug, p.page) for pages in read_pages(PAGES_FILE) for p in pages if p.text}
    configs = {
        "completo": np.ones(len(chunks), dtype=bool),
        "sem-metodologia": outside_methodology(chunks, corpus_methodology()),
    }

    query_vectors = Embedder().queries([q.question for q in questions])
    report: dict = {
        "model": MODEL,
        "questions": len(questions),
        "unreachable": [
            q.id for q in questions if not any((q.slug, p) in text_pages for p in q.pages)
        ],
        "configs": {},
        "per_question": [{"id": q.id, "question": q.question} for q in questions],
    }
    per_config: dict[str, dict[str, np.ndarray]] = {}
    for name, allowed in configs.items():
        top = search(query_vectors, docs, allowed, TOP)
        ranks = []
        for q, row, entry in zip(questions, top, report["per_question"], strict=True):
            rank = first_hit([is_relevant(chunks[i], q, text_pages) for i in row])
            ranks.append(rank)
            entry[name] = {"rank": rank, "top5": [chunks[i]["id"] for i in row[:5]]}
        per_config[name] = metrics(ranks)
        report["configs"][name] = {"chunks": int(allowed.sum()), **summarize(per_config[name])}

    a, b = per_config["sem-metodologia"], per_config["completo"]
    report["diff"] = summarize({name: a[name] - b[name] for name in a})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print_report(report)


def print_report(report: dict) -> None:
    names = list(report["configs"])
    print(f"{report['questions']} perguntas, {len(report['unreachable'])} só em imagem\n")
    print(f"{'':18}" + "".join(f"{n:>26}" for n in names) + f"{'diferença':>26}")
    for metric in report["diff"]:
        cells = [report["configs"][n][metric] for n in names] + [report["diff"][metric]]
        row = "".join(
            f"{c['mean']:>10.3f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]".rjust(26)
            for c in cells
        )
        print(f"{metric:18}{row}")
    misses = [q for q in report["per_question"] if q["completo"]["rank"] is None]
    print(f"\nFora do top {TOP} no índice completo ({len(misses)}):")
    for q in misses:
        print(f"  {q['id']:6} {q['question']}")
    print(f"\nOutput: {OUT_FILE}")


if __name__ == "__main__":
    main()
