"""Extract per-page text from the corpus PDFs into data/processed/pages.jsonl."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/pages.jsonl")

# Header and footer bands, as a fraction of the page height. Body text reaches
# 0.93 of the height in some documents, so position alone does not identify a
# footer: a block is dropped only when it is in a band and also repeats across pages.
TOP_BAND = 0.08
BOTTOM_BAND = 0.88
REPEAT_SHARE = 0.5

# Glyphs from the Symbol and Wingdings fonts, extracted as private-use code points.
PRIVATE_USE = str.maketrans(
    {
        "": "•",
        "": "•",
        "": "•",
        "": "-",
        "": "α",
        "": "®",
    }
)


@dataclass(frozen=True)
class Page:
    slug: str
    page: int
    text: str


@dataclass(frozen=True)
class Block:
    text: str
    in_band: bool


def clean_block(text: str) -> str:
    """Join the wrapped lines of a block into one line.

    A line ending in a hyphen is joined without a space: in this corpus a hyphen at
    the end of a line belongs to a compound word ("deve-se", "anti-inflamatórios"),
    never to a syllable break.
    """
    out = ""
    for line in text.translate(PRIVATE_USE).splitlines():
        line = line.strip()
        if not line:
            continue
        sep = "" if not out or out.endswith("-") else " "
        out += sep + line
    return out


def shape(text: str) -> str:
    """Key used to find repeated blocks: page numbers vary, so digits are masked."""
    return re.sub(r"\d+", "#", text)


def read_blocks(page: pymupdf.Page) -> list[Block]:
    height = page.rect.height
    blocks = []
    for _, y0, _, y1, text, _, block_type in page.get_text("blocks"):
        if block_type != 0:
            continue
        text = clean_block(text)
        if text:
            in_band = y1 < TOP_BAND * height or y0 > BOTTOM_BAND * height
            blocks.append(Block(text, in_band))
    return blocks


def extract_pdf(path: Path, slug: str) -> list[Page]:
    with pymupdf.open(path) as doc:
        pages = [read_blocks(page) for page in doc]
    counts = Counter(shape(b.text) for blocks in pages for b in {b for b in blocks if b.in_band})
    repeated = {s for s, n in counts.items() if n >= REPEAT_SHARE * len(pages)}
    return [
        Page(
            slug=slug,
            page=number,
            text="\n".join(b.text for b in blocks if not (b.in_band and shape(b.text) in repeated)),
        )
        for number, blocks in enumerate(pages, start=1)
    ]


def main() -> None:
    manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = empty = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for entry in manifest:
            pages = extract_pdf(RAW_DIR / f"{entry['slug']}.pdf", entry["slug"])
            for page in pages:
                out.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
            blank = sum(1 for p in pages if not p.text)
            total, empty = total + len(pages), empty + blank
            print(f"{entry['slug']:24} {len(pages):4} pages, {blank} empty")
    print(f"\n{total} pages, {empty} empty. Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
