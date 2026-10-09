# pcdts-rag

RAG sobre os Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) publicados pelo Ministério da Saúde.

A ideia é medir cada etapa do pipeline contra um conjunto de perguntas anotadas à mão antes de decidir o que fica, em vez de avaliar as respostas no olho.

> Projeto de estudo. Não é dispositivo médico e não deve ser usado para orientar conduta clínica. O corpus contém apenas documentos públicos da Conitec.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pcdt-download   # PDFs em data/raw/ e sha256 em data/raw/manifest.json
uv run pytest
```

## Corpus

15 PCDTs vigentes, 1.420 páginas no total, listados em `corpus/sources.toml`. Os PCDTs são atualizados de tempos em tempos, por isso o `data/raw/manifest.json` guarda o sha256 de cada PDF baixado.
