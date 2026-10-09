# ADR 0001: escopo, stack e princípios

- **Status:** aceito
- **Data:** 2026-10-09

## Contexto

Sistemas de RAG costumam ser avaliados de forma qualitativa, o que impede comparar alternativas e detectar regressões. Este projeto adota a avaliação quantitativa como critério de decisão: um componente só é incorporado quando sua contribuição é medida. Isso exige reprodutibilidade do corpus, anotações independentes da implementação e código cujo comportamento seja inspecionável.

## Decisões

**Domínio: PCDTs do SUS.** São documentos públicos, em português, extensos, com tabelas de posologia, critérios de inclusão e exclusão e fluxogramas. O custo de uma resposta incorreta é alto, o que torna citação e recusa requisitos do sistema.

**Python 3.13 e uv.** O Python 3.13 é a versão mais recente com suporte estável nas bibliotecas de ML utilizadas. O uv gerencia ambiente, lockfile e scripts.

**PostgreSQL com pgvector.** Vetores, texto (busca textual na fase 2) e metadados ficam no mesmo banco, com filtros em SQL. Para um corpus de dezenas de documentos, um banco vetorial dedicado não traz benefício que justifique a infraestrutura adicional.

**Sem frameworks de orquestração no núcleo.** Chunking, recuperação, fusão de rankings e o harness de avaliação são implementados diretamente, sem LangChain ou LlamaIndex, para que cada etapa seja testável e mensurável isoladamente. Bibliotecas de escopo restrito são usadas normalmente: cliente pgvector, sentence-transformers e SDKs dos provedores de modelos.

**Relevância anotada por documento e página.** O golden set registra as páginas que contêm a evidência de cada resposta, e não identificadores de chunk. Um chunk é considerado relevante quando cobre uma página anotada. Com isso, a mesma anotação avalia qualquer estratégia de chunking, enquanto anotações por chunk seriam invalidadas a cada alteração na segmentação.

**Corpus versionado por hash.** O `manifest.json` registra o sha256 de cada documento. Os PCDTs são atualizados periodicamente, e resultados só são comparáveis sobre o mesmo manifest.

## Consequências

- Maior volume de código próprio em comparação com o uso de um framework, em troca de controle e observabilidade de cada etapa.
- A anotação por página perde precisão quando a evidência ocupa uma fração pequena de uma página extensa. Esse efeito será reavaliado na fase 2.
