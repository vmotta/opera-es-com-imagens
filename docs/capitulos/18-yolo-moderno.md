# 18. YOLO moderno e inferência prática

## Responsabilidade celular

Na explicação clássica, uma grade divide a imagem e a célula que contém o centro do objeto assume responsabilidade por ele. Isso distribui alvos sem recortar a imagem em janelas independentes.

![Grade YOLO com centro do objeto na célula responsável](../assets/resultados/cap18/01_responsabilidade_celular.png)

Versões modernas podem ser *anchor-free*, usar múltiplas escalas e regras sofisticadas de atribuição. A grade continua útil para compreender que previsões estão ligadas a posições em mapas de features, mas não deve ser confundida com todos os detalhes de uma implementação específica.

## Cabeças multiescala

Objetos pequenos ocupam poucas células em mapas reduzidos. Detectores combinam features em diferentes resoluções: mapas finos preservam localização de objetos pequenos; mapas profundos trazem contexto e semântica para objetos maiores.

## Letterbox e transformação inversa

Redimensionar uma imagem não quadrada diretamente para `640 × 640` deforma objetos. *Letterbox* preserva proporção e adiciona bordas. As caixas previstas pertencem ao espaço preparado; para retornar ao original, removemos padding e dividimos pela escala.

Bibliotecas como Ultralytics encapsulam esse passo, mas compreender o sistema de coordenadas é indispensável ao integrar resultados a outra aplicação.

## Passo a passo do exemplo

O [código do capítulo 18](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap18_yolo_moderno.py):

1. desenha uma grade `7 × 7`, uma caixa e seu centro;
2. destaca a célula responsável;
3. implementa IoU no formato `(x1,y1,x2,y2)`;
4. funciona sem rede e sem internet;
5. quando `--imagem` é fornecida, carrega Ultralytics e executa inferência real;
6. salva a imagem anotada sem abrir janela.

Execução autocontida:

```bash
python -m exemplos.cap18_yolo_moderno
```

Inferência opcional:

```bash
python -m pip install -e ".[yolo]"
python -m exemplos.cap18_yolo_moderno --imagem caminho/rua.jpg --modelo yolov8n.pt
```

O primeiro uso pode baixar pesos. Verifique licença, origem e política de rede do ambiente.

## Interpretando o objeto `Results`

Cada caixa oferece coordenadas, confiança e classe. Converta tensores para CPU/NumPy somente quando necessário; transferências repetidas entre GPU e CPU custam tempo. Para vídeo, processe em lote ou use modo de streaming quando apropriado.

## Avaliação de detectores

- precisão: fração das detecções que estavam corretas;
- revocação: fração dos objetos reais encontrados;
- AP: área sob curva precisão–revocação por classe/IoU;
- mAP: média das APs;
- latência e throughput: tempo por entrada e volume por tempo.

Não compare apenas FPS sem hardware, tamanho de entrada, batch e precisão. O melhor detector é o que satisfaz restrições da aplicação, não o maior número isolado.

## Exercícios

1. Calcule IoU de caixas idênticas, separadas e parcialmente sobrepostas.
2. Compare modelos nano e small em latência e quantidade de detecções no mesmo conjunto.
3. Construa um caso com objeto pequeno e teste duas resoluções de entrada; registre custo e resultado.
