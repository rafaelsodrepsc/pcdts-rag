from pathlib import Path

import pymupdf
import pytest

from pcdt_rag.extract import clean_block, extract_pdf

HEIGHT = 842


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    """Four A4 pages with a running header, a page-number footer and body text."""
    path = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        for n in range(1, 5):
            page = doc.new_page(width=595, height=HEIGHT)
            page.insert_text((72, 40), "Ministério da Saúde")
            page.insert_text((72, 300), f"Corpo da página {n}")
            page.insert_text((290, 800), str(n))
        doc[1].insert_text((72, 780), "Linha de corpo perto do rodapé")
        doc.save(path)
    return path


def test_extract_numbers_pages_from_one(pdf: Path) -> None:
    pages = extract_pdf(pdf, "doc")

    assert [p.page for p in pages] == [1, 2, 3, 4]
    assert {p.slug for p in pages} == {"doc"}
    assert "Corpo da página 3" in pages[2].text


def test_extract_drops_repeated_header_and_page_number(pdf: Path) -> None:
    pages = extract_pdf(pdf, "doc")

    for page in pages:
        assert "Ministério da Saúde" not in page.text
        assert page.text.splitlines()[0].startswith("Corpo da página")
    assert pages[3].text == "Corpo da página 4"


def test_extract_keeps_body_text_inside_footer_band(pdf: Path) -> None:
    pages = extract_pdf(pdf, "doc")

    assert "Linha de corpo perto do rodapé" in pages[1].text


def test_clean_block_keeps_compound_word_hyphen() -> None:
    assert clean_block("o tratamento deve-\nse iniciar\n") == "o tratamento deve-se iniciar"


def test_clean_block_maps_symbol_font_glyphs() -> None:
    assert clean_block(" \nanti-TNF \n") == "• anti-TNFα"
