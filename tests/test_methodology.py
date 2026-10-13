from pcdt_rag.extract import Page
from pcdt_rag.methodology import methodology_pages


def doc(*texts: str) -> list[Page]:
    return [Page("doc", n, text) for n, text in enumerate(texts, start=1)]


def test_methodology_appendix_runs_until_next_appendix() -> None:
    pages = doc(
        "1. INTRODUÇÃO",
        "APÊNDICE 1\nMETODOLOGIA DE BUSCA E AVALIAÇÃO DA LITERATURA",
        "tabela de busca",
        "APÊNDICE 2\nHISTÓRICO DE ALTERAÇÕES",
    )

    assert methodology_pages(pages) == {2, 3}


def test_last_methodology_appendix_runs_to_the_end() -> None:
    pages = doc("texto", "ANEXO A\nMetodologia de busca", "estudos", "")

    assert methodology_pages(pages) == {2, 3, 4}


def test_other_appendices_and_inline_mentions_are_kept() -> None:
    pages = doc(
        "ver o Apêndice 1 sobre a metodologia de busca",
        "APÊNDICE 1\nESCALA EDSS",
        "APÊNDICE 2\nMETODOLOGIA DO TRATAMENTO",
    )

    assert methodology_pages(pages) == set()
