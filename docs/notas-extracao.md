# Notas de extração

Observações sobre a extração de texto dos PDFs do corpus (`uv run pcdt-extract`). Os problemas listados aqui não são tratados na fase 1 e orientam o trabalho da fase 2.

## Resultado

1.420 páginas extraídas em cerca de 7 segundos, gravadas em `data/processed/pages.jsonl`. A numeração usada é a do PDF, começando em 1, e é a mesma usada nas anotações do golden set.

## Cabeçalhos e rodapés

Nenhum documento tem cabeçalho textual repetido. O único elemento repetido é o número de página, presente na faixa de rodapé de 9 dos 15 documentos. Na ordem de leitura do PyMuPDF, esse número aparece antes do corpo da página.

A posição não basta para identificar o rodapé: há corpo de texto até 93% da altura da página em alguns documentos. Um bloco é removido somente quando está numa faixa de margem (8% superiores ou 12% inferiores da página) e o seu formato, com dígitos mascarados, se repete em pelo menos metade das páginas do documento. Foram removidos 720 blocos, todos números de página.

## Numeração impressa divergente

Em `dpoc`, as páginas 29 a 56 e 66 a 75 trazem o número impresso duas vezes. Nas páginas 32 a 56, o número impresso (81 a 105) não corresponde à página do PDF, pois o trecho foi incorporado de outro documento. Referências a páginas devem usar sempre a numeração do PDF.

## Hífen no fim da linha

Há 300 quebras de linha logo após um hífen. Todas pertencem a palavras compostas ("deve-se", "anti-inflamatórios", "pré-eclâmpsia"), nenhuma é separação silábica. As linhas são unidas mantendo o hífen e sem espaço. A opção de remoção de hifenização do PyMuPDF não é usada, pois produziria "devese".

## Glifos de fontes Symbol e Wingdings

Alguns caracteres são extraídos como pontos de código de uso privado. Os observados são mapeados explicitamente:

| Código | Ocorrências | Uso no texto | Substituição |
|---|---:|---|---|
| `U+F0B7` | 985 | marcador de lista | `•` |
| `U+F02D` | 194 | marcador de lista | `-` |
| `U+F0A7` | 38 | marcador de lista | `•` |
| `U+F0FC` | 4 | marcador de lista | `•` |
| `U+F061` | 3 | "TNFα" | `α` |
| `U+F0E2` | 3 | "Respimat®" | `®` |

## Páginas sem texto

10 páginas não têm texto extraível:

| Documento | Páginas | Conteúdo |
|---|---|---|
| `esclerose-multipla` | 33, 34 | Escala EDSS, como imagem |
| `espondilite-ancilosante` | 15, 17 | Fluxogramas de tratamento, como imagem |
| `tdah` | 195 a 197 | Gráficos de metanálise do apêndice de metodologia |
| `doenca-de-alzheimer`, `esclerose-multipla`, `epilepsia` | 20, 40, 63 | Páginas em branco |

A escala EDSS e os fluxogramas são conteúdo clínico que o pipeline não recupera. Perguntas cuja evidência está apenas nessas páginas são mantidas no golden set e marcadas nas notas da anotação, para que a falha seja medida. A inclusão de OCR nessas páginas é avaliada na fase 2.

## Tabelas

Tabelas perdem a estrutura de linhas e colunas. Cada célula vira uma ou mais linhas de texto, e em alguns casos cada linha de uma mesma célula vira uma linha separada ("Número", "de", "estudos"). O quadro da página 121 de `dor-cronica`, por exemplo, gera 181 blocos. O efeito na recuperação é medido pelas perguntas do tipo `tabela` do golden set.
