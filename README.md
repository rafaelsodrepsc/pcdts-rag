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
uv run pcdt-eval       # métricas de retrieval em eval/results/retrieval.json
uv run pytest
```

Para testar uma pergunta (o padrão é `hibrida+minilm`; `--pipeline` troca):

```bash
uv run pcdt-search "Qual a dose máxima diária de metformina?"
uv run pcdt-search "Qual a dose máxima diária de metformina?" --pipeline hibrida+bge
```

## Corpus

15 PCDTs vigentes, 1.420 páginas no total, listados em `corpus/sources.toml`. Os PCDTs são atualizados de tempos em tempos, por isso o `data/raw/manifest.json` guarda o sha256 de cada PDF baixado.

## Avaliação

`eval/golden.jsonl` tem perguntas escritas à mão, cada uma anotada com o documento e as páginas que contêm a resposta. A anotação é por página, e não por chunk, para continuar válida quando a estratégia de chunking mudar: um chunk conta como relevante se é do mesmo documento e contém texto de uma das páginas anotadas.

Algumas perguntas têm a resposta em páginas que no PDF são imagem (a tabela da EDSS, por exemplo). Elas ficam no conjunto de propósito: hoje são erros garantidos, e o número só melhora se o pipeline passar a ler essas páginas.

O `pcdt-eval` compara pipelines de busca, todas sobre o mesmo índice de chunks:

| Pipeline | O que faz |
|---|---|
| `vetorial` | `multilingual-e5-base` (ONNX, sem PyTorch), similaridade de cosseno exata |
| `bm25` | BM25 sobre o texto sem acentos, sem stemming (`lexical.py`) |
| `hibrida` | os 50 primeiros de cada uma, combinados por reciprocal rank fusion (k = 60) |
| `hibrida+minilm` | os 20 primeiros da híbrida reordenados pelo cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` |
| `hibrida+bge` | o mesmo com o `bge-reranker-v2-m3` (export int8) |

Cada pipeline roda uma pergunta por vez, e a latência inclui gerar o embedding da pergunta. As métricas saem com intervalo de confiança de 95% por bootstrap sobre as perguntas:

- `hit@k`: fração das perguntas com pelo menos um chunk relevante entre os k primeiros;
- `mrr@10`: média do inverso da posição do primeiro chunk relevante.

A diferença de cada pipeline para a `vetorial` é calculada pergunta a pergunta, e o intervalo vem do bootstrap dessas diferenças. Para rodar só algumas pipelines: `uv run pcdt-eval --pipelines vetorial hibrida`.

### Resultados

59 perguntas, 1 só em imagem (`em-01`). Latência em CPU, por pergunta, medida por `pcdt-eval` (`eval/results/retrieval.json`).

| Pipeline | hit@1 | hit@5 | hit@10 | mrr@10 | p50 |
|---|---|---|---|---|---|
| `vetorial` | 0,356 | 0,729 | 0,797 | 0,508 | 0,15 s |
| `bm25` | 0,373 | 0,644 | 0,763 | 0,481 | < 1 ms |
| `hibrida` | 0,356 | 0,695 | 0,847 | 0,513 | 0,11 s |
| `hibrida+minilm` | 0,458 | 0,780 | 0,881 | 0,593 | 8,8 s |
| `hibrida+bge` | 0,525 | 0,864 | 0,932 | 0,682 | 44,9 s |

Só o `hibrida+bge` melhora sobre a `vetorial` com IC95% pareado fora de zero nas quatro métricas (hit@5: +0,136, [+0,05; +0,22]). O `hibrida+minilm` só tem hit@10 no limite (+0,085, [0,00; +0,17]), e `bm25` e `hibrida` ficam dentro do ruído. Com 59 perguntas, diferenças menores que cerca de 0,1 não se distinguem.

O padrão da CLI é `hibrida+minilm`, o meio-termo entre qualidade e tempo. O `hibrida+bge` é o mais preciso, mas 45 s por pergunta em CPU só serve para uso em lote.

`methodology.py` identifica os apêndices de metodologia de busca na literatura, que somam 59% das páginas do corpus. Tirá-los do índice mudou o hit@5 em +0,034 com IC95% de [0,00; 0,08] (commit `0310016`), então eles continuam indexados; `pcdt-search --sem-metodologia` ainda permite comparar.
