# pcdts-rag

RAG sobre os Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) publicados pelo Ministério da Saúde.

A ideia é medir cada etapa do pipeline contra um conjunto de perguntas anotadas à mão antes de decidir o que fica, em vez de avaliar as respostas no olho.

> Projeto de estudo. Não é dispositivo médico e não deve ser usado para orientar conduta clínica. O corpus contém apenas documentos públicos da Conitec.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pcdt-download   # PDFs em data/raw/ e sha256 em data/raw/manifest.json
uv run pcdt-extract    # texto por página em data/processed/pages.jsonl
uv run pcdt-chunk      # janelas de 500 tokens em data/processed/chunks.jsonl
uv run pcdt-embed      # embeddings dos chunks em data/processed/embeddings.npz
uv run pcdt-eval       # métricas de retrieval em eval/results/baseline.json
uv run pytest
```

## Corpus

15 PCDTs vigentes, 1.420 páginas no total, listados em `corpus/sources.toml`. Os PCDTs são atualizados de tempos em tempos, por isso o `data/raw/manifest.json` guarda o sha256 de cada PDF baixado.

## Avaliação

`eval/golden.jsonl` tem perguntas escritas à mão, cada uma anotada com o documento e as páginas que contêm a resposta. A anotação é por página, e não por chunk, para continuar válida quando a estratégia de chunking mudar: um chunk conta como relevante se é do mesmo documento e contém texto de uma das páginas anotadas.

Algumas perguntas têm a resposta em páginas que no PDF são imagem (a tabela da EDSS, por exemplo). Elas ficam no conjunto de propósito: hoje são erros garantidos, e o número só melhora se o pipeline passar a ler essas páginas.

O `pcdt-eval` usa o `multilingual-e5-base` (ONNX, sem PyTorch) com busca exata por similaridade de cosseno e reporta, com intervalo de confiança de 95% por bootstrap sobre as perguntas:

- `hit@k`: fração das perguntas com pelo menos um chunk relevante entre os k primeiros;
- `mrr@10`: média do inverso da posição do primeiro chunk relevante.

Cada execução compara o índice completo com o índice sem os apêndices de metodologia de busca na literatura (`methodology.py`), que somam 59% das páginas do corpus.
