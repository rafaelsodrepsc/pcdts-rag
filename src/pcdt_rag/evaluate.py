"""Measure the retrieval pipelines against the golden set (eval/golden.jsonl).

Each question is annotated with the pages that answer it. A retrieved chunk is relevant
when it comes from the same document and covers one of those pages with text: a page that
is only an image cannot be retrieved, and the evaluation counts it as a miss.

Every pipeline runs one question at a time, as it would in use, so the latency includes
embedding the question. Each pipeline is compared with "vetorial", the dense baseline, on
the same questions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcdt_rag.chunk import MODEL, read_pages
from pcdt_rag.extract import OUT_FILE as PAGES_FILE
from pcdt_rag.rerank import MODELS
from pcdt_rag.retrieve import CANDIDATES, PIPELINES, RERANK_DEPTH, RRF_K, Retriever, load_chunks

GOLDEN_FILE = Path("eval/golden.jsonl")
OUT_FILE = Path("eval/results/retrieval.json")
BASELINE = "vetorial"
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
    parser = argparse.ArgumentParser(description="Evaluate the retrieval pipelines.")
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    args = parser.parse_args()
    pipelines = [BASELINE, *(p for p in args.pipelines if p != BASELINE)]

    questions = load_golden(GOLDEN_FILE)
    chunks = load_chunks()
    text_pages = {(p.slug, p.page) for pages in read_pages(PAGES_FILE) for p in pages if p.text}
    retriever = Retriever(chunks)
    report: dict = {
        "model": MODEL,
        "rerankers": {name: m.repo for name, m in MODELS.items()},
        "candidates": CANDIDATES,
        "rerank_depth": RERANK_DEPTH,
        "rrf_k": RRF_K,
        "questions": len(questions),
        "unreachable": [
            q.id for q in questions if not any((q.slug, p) in text_pages for p in q.pages)
        ],
        "pipelines": {},
        "per_question": [{"id": q.id, "question": q.question} for q in questions],
    }
    per_pipeline: dict[str, dict[str, np.ndarray]] = {}
    for name in pipelines:
        retriever.run(name, "aquecimento", TOP)  # loads the models outside the timing
        ranks, latencies = [], []
        for q, entry in zip(questions, report["per_question"], strict=True):
            start = time.perf_counter()
            row = retriever.run(name, q.question, TOP)
            latencies.append(1000 * (time.perf_counter() - start))
            rank = first_hit([is_relevant(chunks[i], q, text_pages) for i in row])
            ranks.append(rank)
            entry[name] = {"rank": rank, "top5": [chunks[i]["id"] for i in row[:5]]}
        per_pipeline[name] = metrics(ranks)
        report["pipelines"][name] = {
            **summarize(per_pipeline[name]),
            "latency_ms": {
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
            },
        }
        print(f"{name}: {np.percentile(latencies, 50):.0f} ms p50", flush=True)

    base = per_pipeline[BASELINE]
    report["diff"] = {
        name: summarize({m: per_pipeline[name][m] - base[m] for m in base})
        for name in pipelines
        if name != BASELINE
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print_report(report)


def print_report(report: dict) -> None:
    pipelines = report["pipelines"]
    metric_names = [m for m in next(iter(pipelines.values())) if m != "latency_ms"]
    print(f"\n{report['questions']} perguntas, {len(report['unreachable'])} só em imagem\n")
    print(f"{'':16}" + "".join(f"{m:>9}" for m in metric_names) + f"{'p50 ms':>9}{'p95 ms':>9}")
    for name, values in pipelines.items():
        cells = "".join(f"{values[m]['mean']:>9.3f}" for m in metric_names)
        latency = values["latency_ms"]
        print(f"{name:16}{cells}{latency['p50']:>9.0f}{latency['p95']:>9.0f}")
    print(f"\nDiferença para {BASELINE}, com IC95% pareado:\n")
    print(f"{'':16}" + "".join(f"{m:>22}" for m in metric_names))
    for name, values in report["diff"].items():
        cells = "".join(
            f"{v['mean']:+.3f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]".rjust(22)
            for v in (values[m] for m in metric_names)
        )
        print(f"{name:16}{cells}")
    print(f"\nOutput: {OUT_FILE}")


if __name__ == "__main__":
    main()
