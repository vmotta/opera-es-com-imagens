# 9. Vídeo, movimento e fluxo óptico

## Vídeo é uma sequência com memória

Cada frame é uma imagem, mas o valor de um sistema de vídeo vem da relação temporal. A 30 FPS, o algoritmo possui cerca de 33 ms por quadro para capturar, processar, desenhar e exibir sem acumular atraso.

Uma aplicação em tempo real não é apenas aquela que “eventualmente termina”; ela precisa respeitar um prazo. Reduzir resolução, processar frames alternados e separar captura de inferência são decisões de engenharia.

## Subtração de fundo

Com câmera estática, podemos modelar o fundo e marcar pixels que se desviam. MOG2 representa cada pixel por distribuições que se adaptam no tempo. Ele é útil, mas não entende objetos: sombra, reflexo ou câmera tremendo também geram mudança.

Uma máscara de movimento costuma passar por limiarização, morfologia e contornos antes de produzir caixas.

## Fluxo óptico e constância de brilho

O fluxo óptico estima um campo de deslocamento. Uma hipótese central é que o mesmo ponto mantém intensidade durante um intervalo curto:

\[
I(x,y,t)=I(x+u,y+v,t+1)
\]

Linearizando, obtemos a restrição:

\[
I_xu+I_yv+I_t=0
\]

Há uma equação e duas incógnitas. Lucas–Kanade assume movimento aproximadamente constante em uma janela e combina várias equações locais. Ainda assim, uma borda reta sofre o **problema da abertura**: conseguimos ver movimento perpendicular à borda, mas não necessariamente ao longo dela. Quinas resolvem melhor o sistema; por isso usamos Shi–Tomasi.

## Pirâmides

Deslocamentos grandes violam a aproximação local. `calcOpticalFlowPyrLK` cria versões reduzidas: no nível menor, um deslocamento grande vira pequeno; a estimativa é refinada ao retornar à resolução original.

## Passo a passo do exemplo

O [código do capítulo 9](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap09_fluxo_optico.py):

1. cria dois frames com um objeto deslocado em `(+68, -42)`;
2. converte para cinza;
3. encontra quinas Shi–Tomasi no primeiro frame;
4. estima novas posições por Lucas–Kanade piramidal;
5. conserva apenas status válido e erro aceitável;
6. desenha setas e calcula o deslocamento mediano.

```bash
python -m exemplos.cap09_fluxo_optico
```

![Vetores do fluxo óptico entre as posições anterior e atual](../assets/resultados/cap09/03_vetores_lucas_kanade.png)

## Oclusão e deriva

Um ponto pode desaparecer atrás de outro objeto. O status ajuda, mas não resolve toda correspondência falsa. Uma verificação comum calcula fluxo ida e volta: rastreia `t → t+1` e depois `t+1 → t`; se o ponto não retorna próximo da origem, é descartado.

Rastreadores acumulam erro. Re-detectar features periodicamente e administrar IDs é necessário em sequências longas.

## Exercícios

1. Aumente o deslocamento e reduza `maxLevel`. Identifique quando o rastreamento falha.
2. Implemente a verificação ida–volta.
3. Crie um vídeo curto sintético, calcule velocidade em pixels/frame e converta para pixels/segundo usando FPS.
