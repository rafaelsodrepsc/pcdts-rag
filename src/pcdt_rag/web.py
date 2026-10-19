"""Local web interface: ask a question, pick a pipeline, read the retrieved passages.

uv run pcdt-web [--port 8000]

The server keeps the index and the models in memory, so after the first search of each
pipeline a question costs what `pcdt-eval` reports instead of a full start-up.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import unicodedata
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from pcdt_rag.chunk import read_pages
from pcdt_rag.corpus import SOURCES_FILE, load_sources
from pcdt_rag.evaluate import OUT_FILE as RESULTS_FILE
from pcdt_rag.extract import OUT_FILE as PAGES_FILE
from pcdt_rag.lexical import BM25, tokenize
from pcdt_rag.retrieve import (
    CANDIDATES,
    DEFAULT_PIPELINE,
    PIPELINES,
    RERANK_DEPTH,
    Retriever,
    load_chunks,
    snippet,
)

PAGE = Path(__file__).parent / "static" / "index.html"
HOSTS = {"127.0.0.1", "localhost"}
MAX_QUERY = 500
MAX_K = 20
PASSAGE = 450
# A word present in more than this share of the chunks says little about the answer.
COMMON = 0.05
# The page has one inline script and one inline style, talks only to this server, and loads
# its two typefaces from Google Fonts (it falls back to the system font when offline).
CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; connect-src 'self'; "
    "style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


def marks(text: str, query: str, bm25: BM25) -> list[list[int]]:
    """[start, end) offsets in `text` of the query words that are rare in the corpus."""
    terms = {
        t
        for t in tokenize(query)
        if t in bm25.weights and len(bm25.weights[t][0]) <= COMMON * bm25.size
    }
    if not terms:
        return []
    # One base letter per character, so an offset in `folded` is the same offset in `text`.
    folded = "".join(unicodedata.normalize("NFKD", ch.lower())[0] for ch in text)
    pattern = re.compile(rf"\b(?:{'|'.join(map(re.escape, sorted(terms)))})\b")
    return [[m.start(), m.end()] for m in pattern.finditer(folded)]


def parse_search(params: dict[str, list[str]]) -> tuple[str, str, int]:
    """(query, pipeline, k) from the query string, or a ValueError a user can read."""
    query = " ".join(params.get("q", [""])[0].split())
    pipeline = params.get("pipeline", [DEFAULT_PIPELINE])[0]
    if not query:
        raise ValueError("Escreva uma pergunta.")
    if len(query) > MAX_QUERY:
        raise ValueError(f"A pergunta pode ter no máximo {MAX_QUERY} caracteres.")
    if pipeline not in PIPELINES:
        raise ValueError(f"Método desconhecido: {pipeline}.")
    try:
        k = int(params.get("k", ["5"])[0])
    except ValueError:
        raise ValueError("k deve ser um número inteiro.") from None
    return query, pipeline, max(1, min(k, MAX_K))


def search_response(
    retriever: Retriever, titles: dict[str, str], query: str, pipeline: str, k: int
) -> dict:
    start = time.perf_counter()
    rows = retriever.run(pipeline, query, k)
    elapsed = time.perf_counter() - start
    results = []
    for i in rows:
        chunk = retriever.chunks[i]
        text = snippet(chunk["text"], query, retriever.bm25, PASSAGE)
        full = " ".join(chunk["text"].split())
        results.append(
            {
                "slug": chunk["slug"],
                "title": titles.get(chunk["slug"], chunk["slug"]),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "text": text,
                "marks": marks(text, query, retriever.bm25),
                "full": full,
                "full_marks": marks(full, query, retriever.bm25),
            }
        )
    return {"pipeline": pipeline, "ms": round(1000 * elapsed), "results": results}


def load_metrics(path: Path) -> dict:
    """Questions, and hit@5 and median latency per pipeline, from the last `pcdt-eval`.

    It is {} when the evaluation has not run, and the page then leaves the numbers out.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return {
            "questions": report["questions"],
            "pipelines": {
                name: {"hit5": p["hit@5"]["mean"], "p50_ms": p["latency_ms"]["p50"]}
                for name, p in report["pipelines"].items()
            },
        }
    except (OSError, KeyError, ValueError):
        return {}


def site_config(chunks: list[dict], titles: dict[str, str], pages: int, metrics: dict) -> dict:
    """What the page shows before the first search."""
    return {
        "pipelines": PIPELINES,
        "default": DEFAULT_PIPELINE,
        "documents": [{"slug": slug, "title": title} for slug, title in titles.items()],
        "pages": pages,
        "chunks": len(chunks),
        "candidates": CANDIDATES,
        "rerank_depth": RERANK_DEPTH,
        "evaluation": metrics,
    }


class App(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], retriever: Retriever, titles: dict[str, str], pages: int
    ):
        super().__init__(address, Handler)
        self.retriever = retriever
        self.titles = titles
        self.config = site_config(retriever.chunks, titles, pages, load_metrics(RESULTS_FILE))
        # The ONNX sessions and the lazy loading of the rerankers are used one search at a time.
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: App

    def do_GET(self) -> None:
        if self.headers.get("Host", "").rsplit(":", 1)[0] not in HOSTS:
            return self.reply_json(HTTPStatus.FORBIDDEN, {"error": "Host não permitido."})
        # Browsers say where a request comes from. Another site must not start searches here.
        if self.headers.get("Sec-Fetch-Site", "none") not in {"same-origin", "none"}:
            return self.reply_json(HTTPStatus.FORBIDDEN, {"error": "Origem não permitida."})
        url = urlsplit(self.path)
        if url.path == "/":
            self.reply(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/config":
            self.reply_json(HTTPStatus.OK, self.server.config)
        elif url.path == "/api/search":
            self.search(parse_qs(url.query))
        else:
            self.reply_json(HTTPStatus.NOT_FOUND, {"error": "Não encontrado."})

    def search(self, params: dict[str, list[str]]) -> None:
        try:
            query, pipeline, k = parse_search(params)
        except ValueError as error:
            return self.reply_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        try:
            with self.server.lock:
                body = search_response(
                    self.server.retriever, self.server.titles, query, pipeline, k
                )
        except Exception as error:  # a failed model download must not kill the server
            self.log_error("search failed: %r", error)
            message = "A busca falhou. Veja o terminal onde o servidor está rodando."
            return self.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": message})
        self.reply_json(HTTPStatus.OK, body)

    def reply_json(self, status: HTTPStatus, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.reply(status, data, "application/json; charset=utf-8")

    def reply(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Carregando o índice e os modelos...", flush=True)
    retriever = Retriever(load_chunks())
    retriever.run(DEFAULT_PIPELINE, "aquecimento", 1)  # loads the default models up front
    titles = {s.slug: s.title for s in load_sources(SOURCES_FILE)}
    pages = sum(len(doc) for doc in read_pages(PAGES_FILE))
    with App(("127.0.0.1", args.port), retriever, titles, pages) as server:
        print(f"Pronto: http://127.0.0.1:{args.port}  (Ctrl+C para sair)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
