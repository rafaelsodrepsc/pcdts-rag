"""Find the literature-search methodology appendices of each PCDT.

Most PCDTs end with appendices describing how the literature was searched and graded.
They are long and say nothing about treatment, so they are a candidate for removal from
the index. This module only finds them; the evaluation decides whether removing them helps.
"""

from __future__ import annotations

import re

from pcdt_rag.extract import Page

# An appendix starts on a line that begins with its heading.
HEADING = re.compile(r"^\s*(AP[ÊE]NDICE|ANEXO)\b", re.MULTILINE)
# How far after the heading to look for the words that identify a methodology appendix.
TITLE_WINDOW = 300


def is_methodology(title: str) -> bool:
    title = title.upper()
    return "METODOLOGIA" in title and ("BUSCA" in title or "LITERATURA" in title)


def methodology_pages(pages: list[Page]) -> set[int]:
    """Pages that belong to a methodology appendix of one document.

    An appendix runs from the page with its heading up to the page before the next
    appendix heading, or to the end of the document.
    """
    starts: dict[int, bool] = {}
    for page in pages:
        for match in HEADING.finditer(page.text):
            title = page.text[match.start() : match.start() + TITLE_WINDOW]
            starts[page.page] = starts.get(page.page, False) or is_methodology(title)
    ordered = sorted(starts.items())
    last = pages[-1].page
    out: set[int] = set()
    for i, (start, method) in enumerate(ordered):
        end = ordered[i + 1][0] - 1 if i + 1 < len(ordered) else last
        if method:
            out.update(range(start, end + 1))
    return out
