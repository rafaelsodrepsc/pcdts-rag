import re

import pytest

from pcdt_rag.chunk import chunk_document
from pcdt_rag.extract import Page


def words(text: str) -> list[tuple[int, int]]:
    """One token per word, enough to test the windowing."""
    return [m.span() for m in re.finditer(r"\S+", text)]


def doc(*texts: str) -> list[Page]:
    return [Page("doc", n, text) for n, text in enumerate(texts, start=1)]


def test_consecutive_chunks_share_overlap_tokens() -> None:
    chunks = chunk_document(doc("w0 w1 w2 w3 w4 w5 w6 w7 w8 w9"), words, size=4, overlap=1)

    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [c.id for c in chunks] == ["doc:0", "doc:1", "doc:2"]


def test_last_window_covers_the_tail() -> None:
    chunks = chunk_document(doc("w0 w1 w2 w3 w4"), words, size=4, overlap=1)

    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4"]


def test_chunk_crosses_page_break_and_records_page_range() -> None:
    chunks = chunk_document(doc("a b c", "d e f"), words, size=4, overlap=0)

    first, second = chunks
    assert (first.page_start, first.page_end, first.text) == (1, 2, "a b c\nd")
    assert (second.page_start, second.page_end, second.text) == (2, 2, "e f")


def test_empty_pages_are_skipped_in_page_range() -> None:
    chunks = chunk_document(doc("a b", "", "c d"), words, size=4, overlap=0)

    assert [(c.page_start, c.page_end, c.text) for c in chunks] == [(1, 3, "a b\nc d")]


def test_text_is_sliced_from_the_original_page() -> None:
    chunks = chunk_document(doc("dose:  7,5 mg/semana"), words, size=10, overlap=0)

    assert chunks[0].text == "dose:  7,5 mg/semana"


def test_document_without_text_has_no_chunks() -> None:
    assert chunk_document(doc("", ""), words, size=4, overlap=1) == []


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_document(doc("a"), words, size=4, overlap=4)
