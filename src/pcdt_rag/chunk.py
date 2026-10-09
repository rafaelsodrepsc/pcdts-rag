"""Split the extracted pages into fixed-size token windows (data/processed/chunks.jsonl)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path

from pcdt_rag.extract import OUT_FILE as PAGES_FILE
from pcdt_rag.extract import Page

MODEL = "intfloat/multilingual-e5-base"
OUT_FILE = Path("data/processed/chunks.jsonl")
# e5 accepts 512 tokens: 500 of text, 2 of the "passage: " prefix and 2 special tokens.
SIZE = 500
OVERLAP = 50

# Maps a text to the (start, end) character offsets of its tokens.
Tokenize = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    id: str
    slug: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class Token:
    page: int
    start: int
    end: int


def chunk_document(
    pages: list[Page], tokenize: Tokenize, size: int = SIZE, overlap: int = OVERLAP
) -> list[Chunk]:
    """Slide a window of `size` tokens over the whole document, crossing page breaks.

    The chunk text is sliced from the page text using the token offsets, so it keeps
    the original characters. Pages inside one chunk are joined with a newline.
    """
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size): {overlap=} {size=}")
    slug = pages[0].slug
    texts = {p.page: p.text for p in pages}
    tokens = [Token(p.page, s, e) for p in pages for s, e in tokenize(p.text)]
    chunks = []
    for start in range(0, max(len(tokens) - overlap, 1), size - overlap):
        window = tokens[start : start + size]
        if not window:
            break
        parts = []
        for page, group in groupby(window, lambda t: t.page):
            group = list(group)
            parts.append(texts[page][group[0].start : group[-1].end])
        chunks.append(
            Chunk(
                id=f"{slug}:{len(chunks)}",
                slug=slug,
                page_start=window[0].page,
                page_end=window[-1].page,
                text="\n".join(parts),
            )
        )
    return chunks


def read_pages(path: Path) -> Iterable[list[Page]]:
    with path.open(encoding="utf-8") as f:
        pages = [Page(**json.loads(line)) for line in f]
    for _, group in groupby(pages, lambda p: p.slug):
        yield list(group)


def e5_tokenizer() -> Tokenize:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(MODEL)
    return lambda text: tokenizer.encode(text, add_special_tokens=False).offsets


def main() -> None:
    tokenize = e5_tokenizer()
    total = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for pages in read_pages(PAGES_FILE):
            chunks = chunk_document(pages, tokenize)
            for chunk in chunks:
                out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"{pages[0].slug:24} {len(chunks):4} chunks")
    print(f"\n{total} chunks ({SIZE} tokens, overlap {OVERLAP}). Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
