from pathlib import Path

import pytest

from pcdt_rag.corpus import Source, download, load_sources

PDF_BYTES = b"%PDF-1.7 fake content"


def write_sources(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sources.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_sources_parses_entries(tmp_path: Path) -> None:
    path = write_sources(
        tmp_path,
        '[[pcdt]]\nslug = "dor-cronica"\ntitle = "Dor Crônica"\nurl = "https://x/a.pdf"\n',
    )

    assert load_sources(path) == [Source("dor-cronica", "Dor Crônica", "https://x/a.pdf")]


def test_load_sources_rejects_duplicated_slugs(tmp_path: Path) -> None:
    entry = '[[pcdt]]\nslug = "a"\ntitle = "A"\nurl = "https://x/a.pdf"\n'
    path = write_sources(tmp_path, entry * 2)

    with pytest.raises(ValueError, match="duplicated"):
        load_sources(path)


def test_download_writes_pdf_and_hashes_it(tmp_path: Path) -> None:
    entry = download(Source("a", "A", "https://x/a.pdf"), tmp_path, fetch=lambda _: PDF_BYTES)

    assert (tmp_path / "a.pdf").read_bytes() == PDF_BYTES
    assert len(entry.sha256) == 64


def test_download_rejects_html_disguised_as_pdf(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="did not return a PDF"):
        download(Source("a", "A", "https://x/a.pdf"), tmp_path, fetch=lambda _: b"<html>")

    assert not (tmp_path / "a.pdf").exists()


def test_download_skips_existing_file(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(PDF_BYTES)

    def fail(_: str) -> bytes:
        raise AssertionError("should not fetch")

    entry = download(Source("a", "A", "https://x/a.pdf"), tmp_path, fetch=fail)

    assert entry.slug == "a"
