# 3. Filtragem, convolução e bordas

## A vizinhança muda o pixel

Nos capítulos anteriores, um pixel mudou de posição. Agora seu valor será recalculado a partir dos vizinhos. Um **kernel** é uma pequena matriz de pesos que desliza sobre a imagem. Para cada posição, multiplicamos valores correspondentes e somamos os produtos.

\[
G(y,x)=\sum_{i=-k}^{k}\sum_{j=-k}^{k}K(i,j)I(y-i,x-j)
\]

Pense no kernel como uma enquete local: cada vizinho dá um voto, e o peso define quanto esse voto vale. Um filtro de média dá votos iguais; um Gaussiano valoriza o centro; um detector de borda dá sinais positivos a um lado e negativos ao outro.

## Suavização não é uma única técnica

| Filtro | Como decide | Vantagem | Custo/limitação |
|---|---|---|---|
| média | média aritmética | simples e rápido | borra bordas e é sensível a extremos |
| Gaussiano | média ponderada | reduz ruído aproximadamente Gaussiano | ainda mistura lados de uma borda |
| mediana | valor central ordenado | excelente contra sal e pimenta | não é convolução linear; custa mais |
| bilateral | espaço + semelhança de intensidade | preserva bordas | custo computacional maior |

Escolher o filtro exige formular um modelo do ruído. Um pixel branco isolado é um valor extremo; a mediana o ignora melhor do que a média. Já pequenas variações distribuídas são tratadas naturalmente pelo Gaussiano.

## Derivadas e bordas

Uma borda é uma mudança rápida de intensidade. O Sobel estima derivadas em duas direções:

- `Sobel X`: responde a mudanças ao longo de `x`, destacando bordas aproximadamente verticais;
- `Sobel Y`: responde a mudanças ao longo de `y`, destacando bordas aproximadamente horizontais.

A derivada pode ser negativa. Por isso, calcular diretamente em `uint8` destruiria metade da informação. O exemplo usa `CV_64F`, toma valor absoluto e só então converte para visualização.

O Canny organiza a detecção em etapas: suavização, gradiente, supressão de não máximos e histerese. A histerese utiliza dois limiares: bordas fortes são aceitas; bordas fracas sobrevivem somente quando conectadas a fortes. É como validar uma pista fraca quando ela continua uma estrada já confirmada.

## Passo a passo do exemplo

O [código do capítulo 3](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap03_filtros_bordas.py):

1. cria formas com intensidades diferentes;
2. adiciona ruído sal e pimenta com semente fixa;
3. compara média, Gaussiano e mediana;
4. calcula Sobel X e Y em ponto flutuante;
5. combina as direções;
6. suaviza antes do Canny e aplica limiares `60/150`.

```bash
python -m exemplos.cap03_filtros_bordas
```

![Comparação entre filtros de média, Gaussiano, mediana, Sobel e Canny](../assets/resultados/cap03/painel.png)

## Como interpretar

A mediana remove os impulsos brancos e pretos mantendo fronteiras mais definidas. O filtro de média espalha cada impulso por sua vizinhança, criando manchas. O Sobel gera respostas largas e graduais; o Canny busca linhas finas e conectadas.

## Parâmetros como compromisso

- kernel maior remove estruturas maiores, mas também apaga detalhes;
- sigma Gaussiano maior aumenta a escala da suavização;
- limiares Canny baixos encontram mais bordas e mais ruído;
- limiares altos produzem menos falsos positivos, mas podem quebrar contornos.

Não existe “melhor” valor universal. Meça em imagens representativas e registre a unidade: intensidade, pixels ou proporção.

## Exercícios

1. Troque o ruído sal e pimenta por ruído Gaussiano. Qual filtro vence e por quê?
2. Aplique Canny sem suavização e conte componentes desconectados.
3. Crie manualmente um kernel de realce com centro positivo e vizinhos negativos. Verifique a soma dos coeficientes e explique o efeito em uma região constante.
