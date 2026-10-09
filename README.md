# pcdts-rag

Sistema de recuperação e resposta a perguntas sobre os Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) publicados pelo Ministério da Saúde.

O desenvolvimento é orientado por avaliação. Cada componente do pipeline é medido contra um conjunto de perguntas anotadas manualmente antes de ser adotado, e os resultados são registrados junto às decisões de arquitetura. O sistema é avaliado em três dimensões:

- **Recuperação:** os trechos retornados contêm a evidência necessária para responder à pergunta.
- **Fidelidade:** a resposta gerada é sustentada pelos trechos recuperados e cita a fonte.
- **Recusa:** o sistema declara a ausência de informação quando a pergunta não é coberta pelos protocolos.

> Este software não é um dispositivo médico e não deve ser usado para orientar conduta clínica. O corpus é composto exclusivamente por documentos públicos da Conitec.

## Status

| Fase | Escopo | Status |
|---|---|---|
| 1. Baseline de recuperação | Corpus, extração de texto, chunking fixo, busca vetorial, golden set, recall@k e MRR | Em andamento |
| 2. Recuperação | Avaliação de estratégias de chunking, busca híbrida (BM25 + vetorial com RRF), reranking | Planejada |
| 3. Geração | Respostas com citação, recusa, fidelidade, comparação de modelos por qualidade e custo | Planejada |
| 4. Operação | API FastAPI com streaming, tracing, custo e latência por consulta, regressão de avaliação no CI | Planejada |

Especificação da fase atual: [`docs/fase-1.md`](docs/fase-1.md).

## Resultados

Os resultados do baseline serão publicados ao final da fase 1. Cada fase seguinte acrescenta suas variações à mesma tabela, com recall@k, MRR, latência e custo.

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Docker com Compose

## Início rápido

```bash
cp .env.example .env
uv sync
docker compose up -d          # PostgreSQL 17 com pgvector
uv run pcdt-download          # baixa o corpus para data/raw/
uv run pcdt-extract           # extrai o texto por página para data/processed/
uv run pytest
```

## Corpus

15 PCDTs vigentes, organizados em cinco grupos temáticos, com cerca de 1.400 páginas. A composição, os critérios de seleção e as características conhecidas dos documentos estão em [`docs/corpus.md`](docs/corpus.md).

## Estrutura

```
corpus/sources.toml    lista de documentos do corpus
data/raw/              PDFs baixados (fora do controle de versão) e manifest.json com sha256
src/pcdt_rag/          código do pipeline
tests/                 testes automatizados
evals/golden/          conjunto de perguntas anotadas
evals/runs/            resultados de execuções de avaliação (fora do controle de versão)
docs/                  especificações, corpus e decisões de arquitetura
```

## Documentação

- [Corpus](docs/corpus.md)
- [Especificação da fase 1](docs/fase-1.md)
- [Guia de anotação do golden set](evals/golden/README.md)
- [ADR 0001: escopo, stack e princípios](docs/adr/0001-escopo-stack-e-principios.md)
