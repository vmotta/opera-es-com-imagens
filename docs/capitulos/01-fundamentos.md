# 1. Imagem digital e OpenCV

## O que você aprenderá

Ao final, você deverá conseguir explicar por que uma imagem é uma matriz, localizar um pixel sem trocar `x` por `y`, distinguir BGR de RGB, recortar uma região de interesse e aplicar uma máscara binária.

## A imagem como uma tabela de intensidades

Pense em um mosaico formado por pequenos azulejos. Visto de longe, o conjunto parece contínuo; de perto, percebemos peças individuais. Cada peça é um **pixel**. A posição da peça fornece as coordenadas e sua tinta fornece a intensidade ou a cor.

Uma imagem em tons de cinza pode ser escrita como uma função discreta:

\[
I(y,x) \in \{0,1,\ldots,255\}
\]

`0` representa preto, `255` representa branco e os valores intermediários representam cinzas. Em uma imagem colorida de 8 bits, guardamos três intensidades por posição. No OpenCV, a forma da matriz costuma ser `(altura, largura, 3)` e a ordem dos canais é **BGR**: azul, verde e vermelho.

!!! warning "A ordem que evita muitos erros"
    A geometria costuma ser falada como `(x, y)`, mas o acesso à matriz é `imagem[y, x]`: primeiro a linha, depois a coluna. A função `cv2.circle`, por outro lado, recebe o centro como `(x, y)`.

## Amostragem, quantização e tipo de dado

Digitalizar uma cena envolve duas discretizações:

- **amostragem:** decide quantas posições espaciais existirão; está relacionada à resolução;
- **quantização:** decide quantos níveis cada posição poderá assumir; em `uint8`, são 256 níveis.

O tipo `uint8` ocupa um byte e não representa números negativos. Ele também tem limite superior. Uma soma NumPy ingênua pode sofrer retorno modular: `250 + 20` pode virar `14`. `cv2.add` faz **saturação**, mantendo o resultado em `255`. É a diferença entre um hodômetro que volta ao zero e um recipiente que simplesmente permanece cheio.

## Região de interesse: trabalhar somente onde importa

Uma ROI (*Region of Interest*) é um recorte da matriz:

```python
roi = imagem[y1:y2, x1:x2].copy()
```

O limite final não é incluído. Portanto, a largura é `x2 - x1` e a altura é `y2 - y1`. O `.copy()` é importante quando desejamos um bloco independente; sem ele, o fatiamento pode ser apenas uma “janela” para a memória original, e uma alteração na ROI modifica a imagem-fonte.

## Máscara binária: um molde vazado

Imagine cobrir uma parede com um molde de pintura. A tinta passa apenas pelas áreas vazadas. Uma máscara executa a mesma seleção:

- `0` (preto): bloqueia o pixel;
- `255` (branco): permite que o pixel passe.

```python
resultado = cv2.bitwise_and(imagem, imagem, mask=mascara)
```

A máscara tem uma dimensão espacial igual à imagem, mas somente um canal. Ela não “recorta” fisicamente a matriz: preserva o tamanho e zera as regiões bloqueadas.

## Passo a passo do exemplo

O [código completo do capítulo 1](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap01_fundamentos.py) executa este pipeline:

1. cria uma imagem BGR sintética para eliminar dependência de arquivos;
2. consulta `shape`, `dtype` e o pixel central;
3. modifica um bloco por fatiamento vetorizado;
4. recorta uma ROI, inverte suas cores e a cola em outra posição;
5. separa B, G e R, intensifica o vermelho com saturação e recompõe os canais;
6. cria uma máscara circular e aplica a operação `AND`;
7. grava todos os estágios para inspeção.

```bash
python -m exemplos.cap01_fundamentos
```

![Comparação entre imagem original, ROI, alteração de canal, máscara e resultado](../assets/resultados/cap01/painel.png)

## O que observar

- O quadrado vermelho no canto foi alterado sem laços `for`; a operação vetorizada atua no bloco inteiro.
- A ROI invertida preserva dimensões porque só pode ser colada em uma fatia do mesmo tamanho.
- A máscara é branca no centro, mas o resultado preserva as cores originais, não se torna branco.

## Erros comuns

| Sintoma | Causa provável | Verificação |
|---|---|---|
| imagem “azulada” no Matplotlib | BGR foi interpretado como RGB | use `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| ROI vazia | limites invertidos ou fora da imagem | imprima `roi.shape` |
| cor estranha após soma | overflow de `uint8` | use `cv2.add` ou converta para tipo maior |
| `AttributeError` após `imread` | arquivo não foi aberto | teste `if imagem is None` antes de `shape` |

## Exercícios

1. Inverta a máscara com `cv2.bitwise_not`. Preveja a área que ficará visível antes de executar.
2. Crie uma máscara retangular e uma circular; combine-as com `bitwise_or` e `bitwise_and`.
3. Explique por que uma imagem `400 × 600 × 3` possui `720.000` elementos, mas somente `240.000` pixels espaciais.

## Síntese

Uma imagem OpenCV é uma matriz NumPy. Dominar eixos, canais, tipos, ROI e máscara não é apenas introdução: essas mesmas operações reaparecem em segmentação, detecção, OCR, redes neurais e vídeo.
