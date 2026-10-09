"""Download the PCDT PDFs listed in corpus/sources.toml into data/raw/."""

from __future__ import annotations

import hashlib
import json
import tomllib
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SOURCES_FILE = Path("corpus/sources.toml")
RAW_DIR = Path("data/raw")
USER_AGENT = "Mozilla/5.0 (pcdt-rag corpus downloader)"


@dataclass(frozen=True)
class Source:
    slug: str
    title: str
    url: str


@dataclass(frozen=True)
class ManifestEntry:
    slug: str
    url: str
    sha256: str
    downloaded_at: str


def load_sources(path: Path) -> list[Source]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sources = [Source(**entry) for entry in data["pcdt"]]
    duplicated = [slug for slug, n in Counter(s.slug for s in sources).items() if n > 1]
    if duplicated:
        raise ValueError(f"duplicated slugs in {path}: {duplicated}")
    return sources


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download(
    source: Source, dest_dir: Path, fetch: Callable[[str], bytes] = fetch
) -> ManifestEntry:
    target = dest_dir / f"{source.slug}.pdf"
    if target.exists():
        content = target.read_bytes()
    else:
        content = fetch(source.url)
        # gov.br sometimes answers a .pdf URL with an HTML page and status 200.
        if not content.startswith(b"%PDF"):
            raise ValueError(f"{source.slug}: {source.url} did not return a PDF")
        target.write_bytes(content)
    return ManifestEntry(
        slug=source.slug,
        url=source.url,
        sha256=hashlib.sha256(content).hexdigest(),
        downloaded_at=datetime.fromtimestamp(target.stat().st_mtime, UTC).isoformat(),
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries, failures = [], []
    for source in load_sources(SOURCES_FILE):
        try:
            entries.append(download(source, RAW_DIR))
            print(f"ok    {source.slug}")
        except (OSError, ValueError) as exc:
            failures.append(source.slug)
            print(f"FAIL  {source.slug}: {exc}")
    manifest = RAW_DIR / "manifest.json"
    manifest.write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")
    print(f"\n{len(entries)} PDFs, {len(failures)} failures. Manifest: {manifest}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
