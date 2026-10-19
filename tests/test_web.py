import json
import threading
import urllib.error
import urllib.request

import pytest

from pcdt_rag.lexical import BM25
from pcdt_rag.web import (
    MAX_QUERY,
    App,
    load_metrics,
    marks,
    parse_search,
    search_response,
    site_config,
)

CHUNKS = [
    {
        "id": "a:0",
        "slug": "a",
        "page_start": 3,
        "page_end": 4,
        "text": "A dose máxima de metformina é 2.550 mg",
    },
    {"id": "b:0", "slug": "b", "page_start": 7, "page_end": 7, "text": "A dose de insulina varia"},
] + [
    {"id": f"c:{i}", "slug": "c", "page_start": i, "page_end": i, "text": "a dose de levodopa"}
    for i in range(40)
]


class FakeRetriever:
    """Stands in for Retriever: it returns the chunks in order, with a real BM25."""

    def __init__(self) -> None:
        self.chunks = CHUNKS
        self.bm25 = BM25([c["text"] for c in CHUNKS])

    def run(self, pipeline: str, query: str, top: int) -> list[int]:
        return [0, 1][:top]


def test_marks_cover_rare_words_and_ignore_accents_and_common_ones() -> None:
    bm25 = FakeRetriever().bm25
    text = "A dose MÁXIMA de metformina"

    found = [
        text[start:end] for start, end in marks(text, "qual a dose maxima de metformina?", bm25)
    ]

    # "dose" and "a" are in almost every chunk, so only the rare words are marked.
    assert found == ["MÁXIMA", "metformina"]


def test_marks_is_empty_when_no_query_word_is_in_the_corpus() -> None:
    assert marks("qualquer texto", "xyzzy", FakeRetriever().bm25) == []


def test_parse_search_defaults_and_clamps() -> None:
    assert parse_search({"q": ["  dose   de  X "], "k": ["99"]}) == (
        "dose de X",
        "hibrida+minilm",
        20,
    )
    assert parse_search({"q": ["x"], "k": ["0"], "pipeline": ["bm25"]}) == ("x", "bm25", 1)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ["   "]},
        {"q": ["x" * (MAX_QUERY + 1)]},
        {"q": ["x"], "pipeline": ["nao-existe"]},
        {"q": ["x"], "k": ["abc"]},
    ],
)
def test_parse_search_rejects_bad_input(params) -> None:
    with pytest.raises(ValueError):
        parse_search(params)


def test_search_response_names_the_document_and_pages() -> None:
    body = search_response(FakeRetriever(), {"a": "PCDT A (Portaria 1)"}, "metformina", "bm25", 2)

    first, second = body["results"]
    assert (first["title"], first["page_start"], first["page_end"]) == ("PCDT A (Portaria 1)", 3, 4)
    assert second["title"] == "b"  # a slug without a title falls back to the slug
    assert [first["text"][s:e] for s, e in first["marks"]] == ["metformina"]
    assert first["full"] == CHUNKS[0]["text"]
    assert [first["full"][s:e] for s, e in first["full_marks"]] == ["metformina"]


def test_site_config_reports_the_index_it_was_given() -> None:
    config = site_config(CHUNKS, {"a": "PCDT A"}, 50, {})

    assert config["pages"] == 50
    assert config["chunks"] == len(CHUNKS)
    assert config["evaluation"] == {}


def test_load_metrics_reads_the_evaluation_report(tmp_path) -> None:
    report = {
        "questions": 59,
        "pipelines": {"bm25": {"hit@5": {"mean": 0.64}, "latency_ms": {"p50": 1.0}}},
    }
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert load_metrics(path) == {
        "questions": 59,
        "pipelines": {"bm25": {"hit5": 0.64, "p50_ms": 1.0}},
    }


def test_load_metrics_is_empty_without_a_report_or_with_a_broken_one(tmp_path) -> None:
    assert load_metrics(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert load_metrics(broken) == {}


@pytest.fixture
def server():
    app = App(("127.0.0.1", 0), FakeRetriever(), {"a": "PCDT A"}, 50)
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{app.server_port}"
    app.shutdown()
    app.server_close()


def get(url: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_server_serves_the_page_the_config_and_searches(server: str) -> None:
    status, page = get(server + "/")
    assert status == 200 and b"<title>Busca nos PCDTs</title>" in page

    status, body = get(server + "/api/config")
    config = json.loads(body)
    assert status == 200 and config["documents"] == [{"slug": "a", "title": "PCDT A"}]

    status, body = get(server + "/api/search?q=metformina&pipeline=bm25&k=1")
    assert status == 200 and len(json.loads(body)["results"]) == 1


def test_server_answers_bad_requests_in_portuguese(server: str) -> None:
    status, body = get(server + "/api/search?q=")
    assert status == 400 and json.loads(body)["error"] == "Escreva uma pergunta."

    assert get(server + "/nao-existe")[0] == 404


def test_server_refuses_a_foreign_host_header(server: str) -> None:
    status, _ = get(server + "/api/config", {"Host": "evil.example"})
    assert status == 403


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_server_refuses_requests_started_by_other_sites(server: str, site: str) -> None:
    assert get(server + "/api/search?q=x", {"Sec-Fetch-Site": site})[0] == 403


def test_server_accepts_requests_from_its_own_page(server: str) -> None:
    assert get(server + "/api/search?q=metformina", {"Sec-Fetch-Site": "same-origin"})[0] == 200
