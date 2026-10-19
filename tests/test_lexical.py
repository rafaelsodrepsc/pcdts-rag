import numpy as np

from pcdt_rag.lexical import BM25, tokenize
from pcdt_rag.retrieve import rrf, snippet, top_k


def test_tokenize_lowercases_and_drops_accents_and_punctuation() -> None:
    assert tokenize("Dose MÁXIMA: 3.000 mg/dia") == ["dose", "maxima", "3", "000", "mg", "dia"]


def test_bm25_ranks_the_document_with_the_rare_term_first() -> None:
    bm25 = BM25(
        [
            "a dose de metformina é de 2.550 mg por dia",
            "a dose de insulina",
            "a dose de levodopa",
        ]
    )

    scores = bm25.scores("dose máxima de metformina")

    assert scores.argmax() == 0
    assert scores[1] == scores[2]


def test_bm25_gives_zero_to_documents_without_query_terms() -> None:
    bm25 = BM25(["asma grave", "dpoc estável"])

    assert bm25.scores("asma").tolist()[1] == 0
    assert bm25.scores("epilepsia").tolist() == [0, 0]


def test_top_k_skips_excluded_and_zero_scores() -> None:
    scores = np.array([0.0, 3.0, 2.0, 1.0])
    allowed = np.array([True, True, False, True])

    assert top_k(scores, allowed, 3) == [1, 3]


def test_rrf_rewards_documents_found_by_both_rankings() -> None:
    dense = [10, 20, 30]
    lexical = [40, 30, 50]

    fused = rrf([dense, lexical])

    assert fused[0] == 30
    assert set(fused) == {10, 20, 30, 40, 50}


def test_rrf_with_one_ranking_keeps_its_order() -> None:
    assert rrf([[3, 1, 2]]) == [3, 1, 2]


def test_snippet_starts_near_the_rarest_query_term() -> None:
    bm25 = BM25(["dose dose dose", "a dose da pregabalina"])
    text = "Introdução longa. " * 10 + "A pregabalina não foi incorporada."

    out = snippet(text, "dose de pregabalina", bm25, size=40)

    assert out.startswith("...")
    assert "pregabalina não foi" in out


def test_snippet_keeps_accented_text_aligned() -> None:
    bm25 = BM25(["fibromialgia", "outra"])
    text = "ção " * 30 + "fibromialgia tratada"

    assert "fibromialgia tratada" in snippet(text, "fibromialgia", bm25, size=40)
