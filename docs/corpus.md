# Corpus

O corpus é definido em [`corpus/sources.toml`](../corpus/sources.toml) e baixado com `uv run pcdt-download`. O arquivo `data/raw/manifest.json` registra a URL e o sha256 de cada PDF. Resultados de avaliação só são comparáveis quando obtidos sobre o mesmo manifest.

## Composição

Versões publicadas em portaria, verificadas em 2026-10-09.

| Grupo | Slug | Documento | Páginas |
|---|---|---|---:|
| Reumatologia e dor | `dor-cronica` | Portaria Conjunta SAES/SAPS/SECTICS nº 1/2024 | 298 |
| | `artrite-reumatoide` | Portaria Conjunta nº 16/2021 | 152 |
| | `artrite-psoriasica` | Portaria Conjunta nº 37/2026 | 46 |
| | `espondilite-ancilosante` | Portaria Conjunta nº 25/2018 | 17 |
| | `psoriase` | Portaria Conjunta nº 18/2021 | 78 |
| Respiratório | `asma` | Portaria Conjunta SAES/SCTIE nº 43/2026 | 57 |
| | `dpoc` | Portaria Conjunta nº 19/2021 | 75 |
| Neurologia | `doenca-de-parkinson` | Portaria Conjunta SAES/SECTICS nº 16/2025 | 77 |
| | `doenca-de-alzheimer` | Portaria Conjunta SAES/SCTIE nº 27/2025 | 90 |
| | `esclerose-multipla` | Portaria Conjunta nº 1/2022 | 91 |
| | `epilepsia` | Portaria Conjunta nº 17/2018 | 63 |
| Metabolismo | `diabete-melito-1` | Portaria Conjunta nº 17/2019 | 58 |
| | `diabete-melito-2` | Portaria SCTIE/MS nº 13/2026 | 80 |
| | `dislipidemia` | Portaria Conjunta nº 8/2019 | 29 |
| Saúde mental | `tdah` | Portaria Conjunta nº 14/2022 | 209 |
| **Total** | | | **1.420** |

## Critérios de seleção

- Somente o texto do protocolo publicado em portaria. Relatórios de recomendação e consultas públicas foram excluídos, pois incluem centenas de páginas de contribuições externas ao protocolo.
- Documentos agrupados por área terapêutica. Protocolos de um mesmo grupo compartilham medicamentos (biológicos na reumatologia, broncodilatadores no grupo respiratório, insulinas e metformina no metabolismo), o que produz trechos semelhantes concorrendo entre si na recuperação.
- Variedade estrutural: tabelas de posologia, critérios de inclusão e exclusão, fluxogramas e questionários.

## Características conhecidas

- **Apêndices de metodologia.** Todos os 15 documentos incluem o apêndice "Metodologia de busca e avaliação da literatura". Somados, esses apêndices ocupam 842 das 1.420 páginas do corpus (59%, contagem por página a partir do título do apêndice). Os casos extremos são `tdah` (90%) e `dor-cronica` (83%). Esses apêndices são indexados no baseline, e sua remoção é avaliada na fase 2.
- **Múltiplos protocolos por arquivo.** `artrite-reumatoide` contém dois protocolos: Artrite Reumatoide (Anexo 1) e Artrite Idiopática Juvenil (Anexo 2).
- **Versão de `epilepsia`.** A versão disponível é de 2018. Há registro de atualização aprovada em 2025, cujo PDF não estava publicado na data da verificação. Substituto previsto: Esclerose Lateral Amiotrófica (Portaria Conjunta nº 13/2020).
- **Texto extraível.** Nenhum documento é digitalizado como imagem, então não há etapa de OCR.

## Aquisição

As URLs apontam para o sufixo `/@@download/file`. As URLs `.pdf` diretas do portal gov.br respondem com uma página HTML e status 200. Por isso, o downloader valida a assinatura `%PDF` de cada arquivo e rejeita respostas que não sejam PDF.

Arquivos já presentes em `data/raw/` não são baixados novamente. Ao alterar a URL de um documento existente, remova o PDF correspondente antes de executar o download.
