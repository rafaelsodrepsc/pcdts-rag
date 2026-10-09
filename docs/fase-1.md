# Fase 1: baseline de recuperação

## Objetivo

Estabelecer um pipeline de recuperação mínimo, de ponta a ponta, e um harness de avaliação que meça sua qualidade. As fases seguintes são comparadas contra os resultados deste baseline.

## Escopo

| Inclui | Não inclui |
|---|---|
| Aquisição do corpus | Geração de respostas |
| Extração de texto por página | Busca híbrida |
| Chunking de tamanho fixo | Reranking |
| Embeddings e indexação em pgvector | API |
| Busca top-k | |
| Golden set com 60 perguntas | |
| Harness de avaliação | |

## Componentes

### 1. Corpus

**Status:** concluído. Composição e características em [`corpus.md`](corpus.md).

### 2. Extração de texto

**Status:** concluído. Observações em [`notas-extracao.md`](notas-extracao.md).

- Extração por página com PyMuPDF, preservando a numeração do PDF.
- Saída: `data/processed/pages.jsonl`, com registros `{"slug", "page", "text"}`.
- Remoção de cabeçalhos e rodapés repetidos.
- Os apêndices de metodologia são mantidos. A remoção deles é avaliada na fase 2, para que o efeito seja medido em vez de presumido.
- Tabelas não recebem tratamento específico. Os problemas de extração observados são registrados em `docs/notas-extracao.md` e orientam o trabalho da fase 2.

**Critério de aceite:** teste com um PDF de fixture verificando texto e numeração das páginas.

### 3. Chunking

**Status:** concluído (`uv run pcdt-chunk`). O corpus gera 2.049 chunks com mediana de 504 tokens, já contando o prefixo `passage: ` e os tokens especiais (máximo de 505, dentro do limite de 512 do modelo). 37% dos chunks ficam numa única página, 54% em duas e 9% em três ou mais. O caso extremo cobre 9 páginas (`dpoc`, páginas 56 a 64), formadas apenas por legendas de gráficos do apêndice de metodologia. Como a relevância é definida pelo intervalo de páginas, chunks largos tornam a métrica mais permissiva; o efeito é considerado na comparação de estratégias de chunking da fase 2.

- Janelas de aproximadamente 500 tokens com sobreposição de 50, contadas pelo tokenizer do modelo de embedding.
- Cada chunk registra `slug`, `page_start` e `page_end`, que associam o chunk às anotações do golden set (ver ADR 0001).
- Saída: `data/processed/chunks.jsonl`.

**Critério de aceite:** testes de sobreposição e de chunks que atravessam a quebra de página.

### 4. Embeddings e indexação

- Modelo: `intfloat/multilingual-e5-base`, com 768 dimensões, executado localmente.
- A família e5 exige os prefixos `query: ` nas consultas e `passage: ` nos documentos.
- Tabela `chunks(id, slug, page_start, page_end, text, embedding vector(768))`, com índice HNSW e distância de cosseno.
- O tempo de indexação e o tamanho do índice são registrados para comparação com a fase 2.

### 5. Busca

`search(question, k) -> list[Chunk]`: consulta ordenada por distância de cosseno.

### 6. Golden set

60 perguntas, conforme o [guia de anotação](../evals/golden/README.md). A anotação é feita em lotes de 10, com execução da avaliação entre os lotes. Um recall próximo de 100% indica perguntas com vocabulário muito próximo ao do documento, que devem ser reescritas.

### 7. Harness de avaliação

Comando `uv run pcdt-eval`:

1. Lê `evals/golden/golden.jsonl`.
2. Executa `search(question, k=10)` para cada pergunta, exceto as do tipo `sem_resposta`.
3. Considera um chunk relevante quando o `slug` coincide e o intervalo `page_start..page_end` contém alguma página da evidência.
4. Calcula recall@1, recall@5, recall@10 e MRR, no agregado e por tipo de pergunta.
5. Grava `evals/runs/<data>-<config>.json` com a configuração (modelo, chunking, k, sha256 do manifest) e os resultados por pergunta.
6. Imprime a tabela de resultados.

As métricas são cobertas por testes com rankings construídos manualmente.

## Critérios de conclusão

- `uv run pcdt-eval` produz as métricas do baseline.
- Resultados publicados no README.
- Análise de falhas das 10 perguntas com pior desempenho, classificadas por causa: tabela, divergência de vocabulário, informação distribuída em várias páginas ou erro de anotação.
- ADR 0002 com as decisões tomadas durante a fase.

## Dependências previstas

`pymupdf`, `sentence-transformers`, `psycopg[binary]`, `pgvector`. Cada uma é adicionada no momento em que passa a ser usada.
