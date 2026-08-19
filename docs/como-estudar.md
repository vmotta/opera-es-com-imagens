# Como estudar este material

Use um ciclo curto em cada capítulo:

1. Leia o **mapa mental** e diga, com suas palavras, qual é a entrada e qual é a saída da técnica.
2. Antes de executar, faça a **previsão** pedida no texto.
3. Rode o exemplo e abra as imagens gravadas em `outputs/`.
4. Mude um único parâmetro; compare o que mudou e explique por quê.
5. Resolva os exercícios sem olhar o código.

## Regra prática de depuração

Quando o resultado estiver errado, não altere tudo de uma vez. Inspecione o pipeline como uma linha de montagem:

| Parada | O que verificar |
|---|---|
| Entrada | Arquivo existe? Dimensões e canais são os esperados? |
| Preparação | Tipo, escala, ordem de canais e tamanho estão corretos? |
| Transformação | Parâmetros têm unidade e intervalo corretos? |
| Resultado intermediário | A máscara, o mapa ou o vetor parecem plausíveis? |
| Decisão | O limiar foi calibrado para o contexto? |

Guardar resultados intermediários não é desperdício: é como acender luzes em diferentes trechos de um encanamento para descobrir onde o fluxo parou.

## Acessibilidade

A documentação utiliza fonte ampliada e admite zoom do navegador sem quebrar o layout. Use `Ctrl` + `+` (Windows/Linux) ou `Command` + `+` (macOS). Imagens possuem texto alternativo; informações importantes não dependem apenas de cor.
