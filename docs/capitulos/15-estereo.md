# 15. Visão 3D, geometria epipolar e estéreo

## Dois olhos recuperam profundidade

Uma câmera projeta pontos 3D em 2D e perde o eixo de profundidade. Duas câmeras separadas por uma distância conhecida observam o mesmo ponto em posições diferentes. Essa diferença horizontal, após retificação, é a **disparidade**.

Coloque um dedo diante do rosto e alterne o olho aberto: objetos próximos “saltam” mais que os distantes. A visão estéreo automatiza essa triangulação.

## Relação de profundidade

Para câmeras retificadas:

\[
Z=\frac{fB}{d}
\]

- `Z`: profundidade;
- `f`: focal em pixels;
- `B`: baseline em metros;
- `d`: disparidade em pixels.

A relação é inversa. Quando `d` se aproxima de zero, a incerteza em `Z` cresce muito. Estéreo mede melhor em uma faixa compatível com focal, baseline e resolução.

## Geometria epipolar e retificação

Sem retificação, o correspondente de um ponto deve ser buscado ao longo de uma linha epipolar inclinada. A calibração estima parâmetros intrínsecos e extrínsecos; a retificação transforma imagens para alinhar essas linhas horizontalmente. Assim, a busca vira essencialmente 1D.

Usar `StereoSGBM` em imagens não retificadas pode produzir um mapa colorido, mas sem significado métrico confiável.

## Block Matching

O algoritmo compara pequenas janelas ao longo da mesma linha. Textura é necessária: uma parede uniforme oferece muitos candidatos idênticos. Reflexos também violam a constância de aparência entre as câmeras.

Parâmetros importantes:

- `numDisparities`: múltiplo de 16 e grande o suficiente para a faixa;
- `blockSize`: janela maior estabiliza regiões pobres, mas borra limites;
- `uniquenessRatio`: exige que a melhor correspondência se destaque;
- filtros de speckle: removem pequenas ilhas incoerentes.

## Passo a passo do exemplo

O [código do capítulo 15](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap15_visao_estereo.py):

1. cria par texturizado;
2. desloca um círculo em 56 px e um retângulo em 16 px;
3. configura SGBM;
4. divide a saída fixa por 16 para obter pixels;
5. normaliza somente disparidades válidas para visualização;
6. calcula profundidade teórica com focal e baseline.

```bash
python -m exemplos.cap15_visao_estereo
```

![Par estéreo sintético e mapa de disparidade](../assets/resultados/cap15/painel.png)

!!! note "Visualização não é profundidade métrica"
    Normalizar o mapa para `0..255` serve para enxergar contraste, mas destrói a escala física. Para calcular `Z`, use a disparidade em ponto flutuante, calibração e máscara de validade.

## Exercícios

1. Dobre o baseline na fórmula e explique o efeito na faixa mensurável.
2. Remova a textura e observe o mapa.
3. Calcule propagação aproximada do erro: compare a variação de `Z` causada por 1 px em `d=8` e em `d=56`.
