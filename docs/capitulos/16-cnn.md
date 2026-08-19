# 16. Construindo Redes Neurais Convolucionais

## Por que não achatar primeiro

Uma MLP que recebe uma imagem `1000 × 1000` começa com um milhão de entradas por canal e perde a organização espacial ao achatar. CNNs usam conectividade local e pesos compartilhados: o mesmo filtro procura um padrão em toda a imagem.

Isso reduz parâmetros e incorpora uma hipótese útil: uma borda continua sendo borda em diferentes posições.

## Dimensão da convolução

Para entrada `N`, kernel `K`, padding `P` e stride `S`, a saída em um eixo é:

\[
\left\lfloor\frac{N+2P-K}{S}\right\rfloor+1
\]

Com `same`, stride 1 e kernel ímpar, preservamos dimensão. Stride 2 reduz aproximadamente pela metade.

Cada filtro produz um feature map. Com 32 filtros, a saída possui 32 canais, independentemente dos 3 canais de entrada.

## ReLU e pooling

`ReLU(x)=max(0,x)` introduz não linearidade e zera respostas negativas. Não significa que negativas eram “erradas”; cria uma representação que a rede aprende a usar.

Max pooling guarda o maior valor de cada janela e reduz resolução. É como resumir um bairro pelo sinal mais forte. A redução diminui custo e fornece tolerância a pequenos deslocamentos, mas perde localização precisa.

## Da extração à decisão

Blocos convolucionais formam representação hierárquica. `Flatten` transforma mapas em vetor; camadas `Dense` combinam características; Softmax normaliza pontuações em distribuição entre classes mutuamente exclusivas.

Para classificação multirrótulo, use saídas sigmoides independentes — Softmax imporia competição indevida.

## Passo a passo do exemplo

O [código do capítulo 16](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap16_cnn.py) possui duas partes:

1. aplica kernels verticais/horizontais manualmente com `filter2D`;
2. executa ReLU e max pooling NumPy para tornar as operações visíveis;
3. mostra a redução espacial;
4. se TensorFlow estiver instalado, constrói uma CNN `Conv → Pool → Conv → Pool → Dense` e imprime parâmetros.

```bash
python -m exemplos.cap16_cnn
# parte Keras opcional
python -m pip install -e ".[deep]"
```

![Entrada, feature maps, ReLU e max pooling](../assets/resultados/cap16/01_feature_maps.png)

## Parâmetros e generalização

Mais camadas e filtros aumentam capacidade, mas não garantem generalização. Monitore separadamente treino e validação. Augmentation precisa preservar o rótulo: espelhar um dígito ou uma imagem médica pode mudar significado.

Regularização inclui dropout, weight decay, augmentation e parada antecipada. A divisão de dados deve evitar vazamento por pessoa, vídeo ou origem.

## Exercícios

1. Calcule manualmente as dimensões após cada camada do exemplo Keras.
2. Conte parâmetros da primeira Conv2D: inclua bias.
3. Compare classificação multiclasse e multirrótulo, escolhendo ativação e função de perda.
