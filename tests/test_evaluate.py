import tomllib
from pathlib import Path

import numpy as np

from pcdt_rag.embed import mean_pool
from pcdt_rag.evaluate import (
    GOLDEN_FILE,
    Question,
    bootstrap_ci,
    first_hit,
    hit_at,
    is_relevant,
    load_golden,
    reciprocal_rank,
)
from pcdt_rag.retrieve import outside_methodology, search

ROOT = Path(__file__).parent.parent


def chunk(slug: str, start: int, end: int) -> dict:
    return {"slug": slug, "page_start": start, "page_end": end}


QUESTION = Question("q", "asma", (20,), "?", "!")
TEXT_PAGES = {("asma", n) for n in range(1, 40)}


def test_chunk_covering_an_annotated_page_is_relevant() -> None:
    assert is_relevant(chunk("asma", 19, 20), QUESTION, TEXT_PAGES)
    assert is_relevant(chunk("asma", 20, 21), QUESTION, TEXT_PAGES)


def test_chunk_from_other_pages_or_documents_is_not_relevant() -> None:
    assert not is_relevant(chunk("asma", 21, 22), QUESTION, TEXT_PAGES)
    assert not is_relevant(chunk("dpoc", 19, 21), QUESTION, TEXT_PAGES)


def test_page_without_text_cannot_be_hit() -> None:
    """A chunk spanning an image page carries no text from it."""
    image_page = Question("q", "asma", (33,), "?", "!")
    text_pages = TEXT_PAGES - {("asma", 33)}

    assert not is_relevant(chunk("asma", 32, 34), image_page, text_pages)


def test_metrics_from_ranks() -> None:
    ranks = [1, 3, None, 7]

    assert hit_at(ranks, 1).tolist() == [1, 0, 0, 0]
    assert hit_at(ranks, 5).tolist() == [1, 1, 0, 0]
    assert reciprocal_rank(ranks).tolist() == [1, 1 / 3, 0, 1 / 7]
    assert first_hit([False, False, True, True]) == 3
    assert first_hit([False, False]) is None


def test_bootstrap_interval_contains_the_mean_and_is_reproducible() -> None:
    values = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float)

    low, high = bootstrap_ci(values, resamples=2000)

    assert low <= values.mean() <= high
    assert bootstrap_ci(values, resamples=2000) == [low, high]


def test_search_ranks_by_similarity_and_skips_excluded_docs() -> None:
    docs = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    query = np.array([[1, 0]], dtype=np.float32)

    assert search(query, docs, np.array([True, True, True]), 2).tolist() == [[0, 1]]
    assert search(query, docs, np.array([False, True, True]), 2).tolist() == [[1, 2]]


def test_mean_pool_ignores_padding_and_normalizes() -> None:
    hidden = np.array([[[3, 0], [0, 4], [100, 100]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])

    pooled = mean_pool(hidden, mask)

    np.testing.assert_allclose(pooled, [[0.6, 0.8]], rtol=1e-6)


def test_golden_set_is_well_formed() -> None:
    sources = tomllib.loads((ROOT / "corpus/sources.toml").read_text(encoding="utf-8"))
    slugs = {p["slug"] for p in sources["pcdt"]}
    questions = load_golden(ROOT / GOLDEN_FILE)

    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.slug in slugs, q.id
        assert q.pages and all(p > 0 for p in q.pages), q.id
        assert q.question.strip() and q.answer.strip(), q.id


def test_only_chunks_entirely_inside_methodology_are_excluded() -> None:
    methodology = {("asma", 52), ("asma", 53)}
    chunks = [chunk("asma", 51, 52), chunk("asma", 52, 53), chunk("dpoc", 52, 53)]

    assert outside_methodology(chunks, methodology).tolist() == [True, False, True]
