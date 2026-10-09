# Golden set

Perguntas anotadas manualmente em `golden.jsonl`, uma por linha. Meta: 60 perguntas na fase 1 e 150 até a fase 3.

## Formato

```json
{"id": "dor-001", "question": "Qual a dose máxima diária de paracetamol para adultos no PCDT de dor crônica?", "type": "dose", "evidence": [{"slug": "dor-cronica", "pages": [42]}], "answer": "...", "notes": ""}
```

| Campo | Descrição |
|---|---|
| `id` | `<slug>-<número>`. Identificadores são permanentes e não são reutilizados |
| `question` | Pergunta redigida como um profissional de saúde a formularia, sem reproduzir o texto do documento |
| `type` | `factual`, `dose`, `criterio` (inclusão/exclusão), `tabela`, `comparacao` (entre PCDTs) ou `sem_resposta` |
| `evidence` | Documento e páginas (numeração do PDF, a partir de 1) que contêm a resposta. Vazio para `sem_resposta` |
| `answer` | Resposta de referência, curta, utilizada a partir da fase 3 |
| `notes` | Ambiguidades e justificativas de anotação |

## Regras de anotação

- A pergunta é redigida antes da consulta ao documento, para evitar a reprodução do vocabulário do texto, que infla artificialmente os resultados de recuperação.
- No mínimo 15% das perguntas são do tipo `sem_resposta`: plausíveis, mas não cobertas pelos protocolos. Elas avaliam a recusa a partir da fase 3.
- No mínimo 20% das perguntas têm resposta contida em tabela.
- Nenhum documento concentra mais de 20% das perguntas.
- As evidências apontam somente para o conteúdo clínico, nunca para os apêndices de metodologia.
