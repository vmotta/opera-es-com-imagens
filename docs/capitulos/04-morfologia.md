# 4. Limiarização e morfologia matemática

**Da intensidade dos pixels à identificação e ao tratamento de regiões de interesse**

## Apresentação do capítulo

Imagine que você precisa contar comprimidos em uma esteira, separar letras do papel em um documento digitalizado ou medir a área ocupada por sementes em uma fotografia. Embora sejam aplicações diferentes, todas começam com uma pergunta semelhante: **quais partes da imagem devem participar da análise?** Antes de contar ou medir, precisamos distinguir o que interessa daquilo que será tratado como fundo.

Uma fotografia não fornece essa distinção diretamente. Ela registra intensidades e cores: uma região pode ser clara porque pertence a um objeto, porque recebeu mais luz ou porque apresenta um reflexo. Para transformar esses valores em informação útil, precisamos estabelecer critérios. Neste capítulo, começaremos com um critério simples: comparar a intensidade de cada pixel com um valor de referência, chamado **limiar**.

Essa primeira decisão raramente é perfeita. Podem aparecer pontos isolados, pequenos buracos ou interrupções em regiões que deveriam ser contínuas. A morfologia matemática permite modificar essas regiões com base em sua forma e em sua vizinhança. O objetivo não é apenas produzir uma imagem visualmente agradável: é obter uma representação adequada à tarefa, preservando as informações necessárias para contar, reconhecer ou medir.

Ao longo do capítulo, usaremos exemplos de objetos claros sobre fundo escuro e de texto escuro sobre papel claro. A diferença entre esses casos será importante para compreender a polaridade da máscara. Sempre que falarmos em expansão ou redução de objetos, adotaremos, salvo indicação contrária, a convenção de que **o objeto de interesse está branco e o fundo está preto**.

## O que você aprenderá

Ao concluir o estudo, você deverá ser capaz de explicar o que é uma máscara binária, escolher entre limiar fixo, Otsu e limiar adaptativo, interpretar os limites de um histograma e justificar a polaridade adotada. Também deverá compreender como o elemento estruturante orienta erosão, dilatação, abertura, fechamento e gradiente morfológico.

Você aprenderá ainda a relacionar parâmetros à escala da imagem, reconhecer quando uma operação destrói informação válida, usar top-hat e black-hat em imagens de intensidade e organizar um procedimento de segmentação com inspeção das etapas intermediárias. Os exemplos em Python mostram a aplicação prática dessas ideias, enquanto os exercícios ajudam a desenvolver critérios para interpretar os resultados.

**Conhecimentos prévios.** É suficiente reconhecer uma imagem como uma matriz de números e conhecer variáveis, chamadas de funções e importações em Python. As equações serão apresentadas depois da explicação intuitiva. Não é necessário memorizar todas as fórmulas para começar a experimentar.

**Convenção dos exemplos.** Trabalharemos com imagens de um canal, armazenadas como `uint8`, com intensidades inteiras de `0` a `255`. Essa é uma representação comum, mas não a única possível em processamento de imagens. Nas máscaras, usaremos exclusivamente `0` para fundo e `255` para primeiro plano.

---

## 4.1 Da intensidade para uma decisão

### O que há em um pixel?

Em uma imagem em tons de cinza, cada pixel armazena uma intensidade. No intervalo adotado aqui, `0` representa preto, `255` representa branco e os valores intermediários representam diferentes tons de cinza. Um pixel de valor `180` é mais claro que um pixel de valor `60`, mas esses números não dizem, sozinhos, a que objeto cada pixel pertence.

Considere uma fotografia de peças claras sobre uma superfície escura. Se as intensidades das peças forem consistentemente maiores que as do fundo, podemos usar essa diferença para separá-las. Escolhemos um valor de referência e classificamos os pixels de acordo com o lado em que ficam. Essa operação é chamada **limiarização**.

Uma analogia simples é uma regra de aprovação por nota: a nota pode assumir muitos valores, mas a regra produz apenas duas respostas. Da mesma forma, a limiarização reduz vários níveis de intensidade a duas categorias. A analogia ajuda a entender a decisão; ela não significa que o valor escolhido seja necessariamente adequado ao problema.

Para uma imagem de intensidade \(I\), um limiar \(T\) e uma saída binária \(B\), usaremos:

\[
B(y,x)=
\begin{cases}
255, & I(y,x)>T,\\
0, & I(y,x)\leq T.
\end{cases}
\]

O símbolo \(I(y,x)\) significa “intensidade do pixel na linha \(y\) e na coluna \(x\)”. O resultado \(B(y,x)\) é a classificação desse pixel. A expressão não calcula uma média nem reconhece uma forma: ela faz uma comparação.

### Exemplo numérico comentado

Se o limiar for `127`, teremos:

| Intensidade original | Comparação com 127 | Saída |
|---|---|---|
| 30 | 30 não é maior que 127 | 0 — preto |
| 100 | 100 não é maior que 127 | 0 — preto |
| 127 | Igual ao limiar | 0 — preto |
| 128 | 128 é maior que 127 | 255 — branco |
| 220 | 220 é maior que 127 | 255 — branco |

Observe o caso de igualdade: na regra usada aqui, um pixel igual ao limiar fica preto. Essa diferença de um nível pode ser relevante ao interpretar exemplos pequenos e resultados de código.

### Por que chamamos o resultado de máscara?

Uma máscara indica onde uma operação deverá atuar. Se os pixels das peças estiverem brancos e os demais estiverem pretos, podemos usar essa máscara para selecionar as peças, calcular sua área em pixels ou procurar regiões conectadas. A máscara não precisa preservar a aparência original: sua função é representar uma seleção espacial.

**A limiarização descarta informação.** Depois que `128` e `220` se tornam `255`, a máscara já não permite recuperar a diferença entre essas intensidades. Por isso, mantenha a imagem original e armazene a máscara em outra variável. Uma boa prática é usar a imagem original para observar o problema e a máscara para representar a decisão.

---

## 4.2 Limiar fixo: uma regra para a imagem inteira

O limiar fixo utiliza o mesmo valor em todas as posições. Esse método pode funcionar bem quando as condições de aquisição são estáveis: iluminação controlada, fundo previsível e contraste consistente entre objeto e fundo. Sua simplicidade facilita compreender e depurar o processo.

O número `127`, frequente em exemplos introdutórios, não tem um significado especial para a sua aplicação. Ele fica próximo do centro do intervalo de 8 bits, mas o melhor ponto de separação depende das intensidades presentes na imagem. Se o fundo estiver perto de `150` e os objetos perto de `220`, um limiar de `127` classificará ambos como brancos.

### Preparação da imagem

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem = cv2.imread("imagem.png")

if imagem is None:
    raise FileNotFoundError(
        "Não foi possível abrir imagem.png. Confira o nome e a pasta."
    )

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
```

A leitura padrão com `cv2.imread` fornece os canais na ordem BGR. A conversão indicada produz uma imagem de um canal. Se você já leu o arquivo com `cv2.IMREAD_GRAYSCALE`, essa conversão não é necessária.

A verificação de `imagem is None` evita continuar com uma leitura malsucedida. Um erro posterior na conversão pode parecer um problema de processamento, quando a causa real é simplesmente um caminho incorreto.

### Aplicação do limiar

```python
limiar_usado, binaria = cv2.threshold(
    cinza,
    127,
    255,
    cv2.THRESH_BINARY
)
```

Leia essa chamada como uma instrução: “examine `cinza`, use `127` como referência e atribua `255` aos pixels que satisfazem a regra binária”. O valor `255` é o nível de saída para a classe branca; ele não é um segundo limiar.

A função retorna dois elementos. O primeiro registra o limiar utilizado; o segundo contém a imagem produzida. Quando o primeiro não será consultado, é comum escrever `_, binaria = ...`. O sublinhado é apenas uma convenção de programação para indicar um resultado que não será aproveitado.

### O que acontece quando o limiar muda?

Para `THRESH_BINARY`, aumentar o limiar torna mais exigente a condição para um pixel ficar branco. Portanto, mantendo a imagem, a quantidade de pixels brancos não aumenta. Diminuir o limiar facilita a classificação como branco e pode incluir partes do fundo.

Essa observação fornece um diagnóstico prático. Se a máscara quase inteira ficou branca, o limiar pode estar baixo demais para a polaridade escolhida. Se quase tudo ficou preto, pode estar alto demais. Antes de ajustar, confirme se branco realmente representa o que você deseja selecionar.

---

## 4.3 Polaridade: quem representa o objeto?

**Primeiro plano** é a classe de interesse para a operação. **Fundo** é a classe restante. Esses termos indicam papéis na análise, e não uma posição física: o primeiro plano não precisa ser a parte mais próxima da câmera.

Em um documento, as letras geralmente são escuras e o papel é claro. Se usarmos a regra binária convencional, o papel pode ficar branco e as letras pretas. Isso não torna a máscara inválida, mas dificulta interpretar operações morfológicas quando estamos pensando em letras como objetos. Para trabalhar com letras brancas, podemos inverter a decisão:

```python
_, texto_branco = cv2.threshold(
    cinza,
    127,
    255,
    cv2.THRESH_BINARY_INV
)
```

Nesse caso, valores menores ou iguais a `127` tornam-se `255`, e valores maiores tornam-se `0`. Para o mesmo limiar, as saídas de `THRESH_BINARY` e `THRESH_BINARY_INV` são complementares nas máscaras `0/255`.

| Situação | Classe que queremos branca | Polaridade inicial a experimentar |
|---|---|---|
| Peças claras sobre base escura | Peças claras | `THRESH_BINARY` |
| Letras escuras sobre papel claro | Letras escuras | `THRESH_BINARY_INV` |
| Furos escuros em uma superfície clara | Furos escuros | `THRESH_BINARY_INV` |

Essas escolhas são pontos de partida, não garantias de separação. Um reflexo pode ser mais claro que a peça e uma sombra pode ser mais escura que uma letra.

**Antes da morfologia, faça uma pergunta objetiva: os pixels brancos correspondem ao que desejo tratar como objeto?** A dilatação expande as regiões de maior valor. Se o papel estiver branco, ela expandirá o papel e poderá afinar visualmente as letras pretas. O algoritmo não trocou de comportamento; o papel da classe branca é que mudou.

---

## 4.4 Histograma: observar a distribuição antes de decidir

Um histograma de intensidades informa quantos pixels possuem cada valor. O eixo horizontal representa os níveis de cinza; o eixo vertical representa a quantidade de pixels. Ele resume a distribuição tonal da imagem e ajuda a avaliar se uma separação por intensidade parece plausível.

Imagine uma sala com dois grupos de pessoas: um grupo tem alturas concentradas em uma faixa e o outro, em uma faixa distante. Uma referência entre as faixas pode separá-los razoavelmente. Se as alturas se sobrepõem muito, nenhuma referência única produzirá uma divisão perfeita. Com intensidades, ocorre algo semelhante.

Quando há dois grupos predominantes, falamos em uma distribuição **bimodal**. Uma imagem com fundo escuro relativamente uniforme e objetos claros pode apresentar dois picos. Um vale entre eles sugere uma região de possíveis limiares, mas os picos não vêm acompanhados de rótulos como “objeto” e “fundo”.

```python
plt.figure(figsize=(10, 4))
plt.hist(cinza.ravel(), bins=256, range=(0, 256), color="dimgray")
plt.axvline(127, color="red", label="Limiar de referência: 127")
plt.xlabel("Intensidade")
plt.ylabel("Quantidade de pixels")
plt.title("Distribuição das intensidades")
plt.legend()
plt.tight_layout()
plt.show()
```

`ravel()` apresenta os valores da matriz como uma sequência. Isso permite contar as intensidades sem considerar a organização em linhas e colunas.

### O que o histograma não revela

O histograma não mostra onde cada intensidade ocorre. Duas imagens podem ter exatamente as mesmas contagens e arranjos espaciais completamente diferentes. Uma delas pode mostrar uma peça contínua; outra, os mesmos pixels distribuídos aleatoriamente.

Também não é correto concluir que todo histograma com dois picos contém dois objetos. Os picos representam grupos de intensidade, não quantidades de objetos. Uma coleção de cem peças claras sobre uma base escura pode ter apenas dois grupos tonais principais.

Ao analisar uma imagem, observe o histograma junto com a própria cena. Procure sobreposição entre objeto e fundo, sombras, reflexos e diferenças entre regiões. Se os mesmos valores aparecem em classes diferentes, a intensidade isolada poderá ser insuficiente.

---

## 4.5 Método de Otsu: escolha automática de um limiar global

No limiar fixo, alguém escolhe o valor. No método de Otsu, um critério estatístico orienta essa escolha. O procedimento considera divisões possíveis do histograma em duas classes e seleciona uma divisão que favorece a separação entre elas.

A ideia pode ser entendida por uma organização de grupos. Uma divisão ruim coloca intensidades muito diferentes dentro do mesmo grupo. Uma divisão mais favorável produz grupos internamente mais concentrados. A **variância** expressa essa dispersão: ela cresce quando os valores se afastam de sua média.

Para um limiar candidato \(t\), a variância dentro das classes, ponderada pela participação de cada classe, pode ser escrita como:

\[
\sigma_w^2(t)=\omega_0(t)\sigma_0^2(t)+\omega_1(t)\sigma_1^2(t).
\]

Os termos \(\omega_0\) e \(\omega_1\) indicam as proporções de pixels nas duas classes. Os termos \(\sigma_0^2\) e \(\sigma_1^2\) indicam a dispersão das intensidades em cada uma. Otsu procura minimizar essa combinação ponderada. Uma formulação equivalente maximiza a variância entre as classes:

\[
\sigma_b^2(t)=\omega_0(t)\omega_1(t)
\left[\mu_0(t)-\mu_1(t)\right]^2.
\]

Aqui, \(\mu_0\) e \(\mu_1\) são as médias das classes. O termo quadrático expressa a distância entre essas médias, enquanto os pesos consideram a participação dos grupos. Não é necessário implementar as equações para usar o método, mas compreendê-las ajuda a perceber seus limites.

### Aplicação em Python

```python
limiar_otsu, binaria_otsu = cv2.threshold(
    cinza,
    0,
    255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)

print(f"Limiar escolhido por Otsu: {limiar_otsu:.0f}")
```

O operador `|` combina as opções. O `0` ocupa o argumento do limiar, mas a escolha efetiva será feita pelo método. Para selecionar estruturas escuras como primeiro plano, combine `THRESH_BINARY_INV` com `THRESH_OTSU`.

### O que “automático” significa neste caso?

Significa que você não precisa informar manualmente o valor final de corte. Não significa que o método conheça o objeto desejado ou avalie a qualidade da segmentação para a sua tarefa. Uma divisão estatisticamente favorável pode separar uma sombra de uma área iluminada, em vez de separar uma peça do fundo.

Otsu continua usando **um único limiar para toda a imagem**. Ele pode funcionar especialmente bem quando as classes apresentam intensidades suficientemente distintas. Sobreposição, iluminação irregular, reflexos e classes com proporções muito diferentes podem dificultar a correspondência entre a divisão estatística e a divisão desejada.

Um detalhe instrutivo: se uma imagem possui apenas intensidades `70` e `190`, qualquer corte de `70` até `189` produz a mesma máscara com a regra `I > T`. Portanto, o valor retornado não precisa ser o ponto médio `130`. Um resultado longe do centro do intervalo não representa, por si só, um erro.

---

## 4.6 Suavizar antes de limiarizar: benefício e custo

O ruído provoca variações que não representam necessariamente diferenças reais entre as regiões. Uma superfície uniforme pode aparecer com valores como `79`, `82`, `77` e `84`. Perto do limiar, pequenas flutuações podem fazer pixels vizinhos receberem classes diferentes e gerar uma máscara fragmentada.

A suavização reduz parte dessas variações combinando informações da vizinhança. No filtro gaussiano, posições próximas ao centro recebem maior influência. Isso pode tornar as regiões mais regulares antes da classificação.

```python
suave = cv2.GaussianBlur(cinza, (5, 5), 0)

limiar_suave, otsu_suave = cv2.threshold(
    suave,
    0,
    255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
```

A janela `(5, 5)` define a extensão espacial do filtro. Nesse uso, o último `0` permite que o desvio padrão seja determinado a partir do tamanho da janela. O filtro de suavização e o elemento estruturante da morfologia são conceitos diferentes, mesmo quando ambos recebem informalmente o nome de kernel.

### Por que não suavizar sempre com uma janela grande?

Porque bordas e detalhes também são variações de intensidade. Uma linha fina pode perder contraste quando misturada ao fundo. Dois objetos próximos podem ter sua separação menos nítida. A limiarização posterior poderá então apagar a linha ou unir as regiões.

Considere uma letra com traço de dois pixels. Uma janela muito maior que esse traço mistura pixels da letra e do papel, alterando a informação que você deseja preservar. A suavização deve ser avaliada em relação à menor estrutura importante da imagem.

Para ruído impulsivo, como pontos muito claros e muito escuros isolados, um filtro de mediana pode ser uma alternativa a comparar. Isso não dispensa observar o resultado: nenhum filtro distingue automaticamente ruído de um detalhe legítimo apenas porque ele é pequeno.

---

## 4.7 Limiar adaptativo: uma referência para cada vizinhança

Imagine fotografar uma folha perto de uma janela. Um lado fica iluminado e o outro fica sombreado. Um pixel do papel na sombra pode ser mais escuro que parte de uma letra no lado iluminado. Nessas condições, um único valor de corte pode não atender às duas regiões.

A limiarização adaptativa calcula uma referência local. Em vez de comparar todos os pixels com o mesmo número, ela compara cada pixel com um valor obtido de sua vizinhança. A pergunta passa a ser: “este pixel é claro ou escuro em relação à região ao seu redor?”.

Uma expressão simplificada é:

\[
T(y,x)=M(y,x)-C,
\]

em que \(M(y,x)\) é uma média local, simples ou ponderada, e \(C\) é um deslocamento constante. A classificação usa esse limiar variável na posição de cada pixel.

```python
adaptativa = cv2.adaptiveThreshold(
    cinza,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    5
)
```

Neste exemplo, adotamos polaridade invertida para favorecer texto escuro como objeto branco. A vizinhança tem `21 × 21` pixels. A constante `5` é subtraída da referência local.

### Interpretando os parâmetros com um cálculo simples

Suponha que a referência local seja `180` e `C = 5`. O limiar local será `175`. Um pixel de valor `120` será branco na regra invertida, pois está abaixo desse limiar; um pixel de valor `185` será preto.

Se aumentarmos `C` para `15`, o limiar local cairá para `165`. Na polaridade invertida, a condição para ficar branco se torna mais restritiva: pixels entre `166` e `175`, antes selecionados, deixam de ser selecionados. Na polaridade convencional, reduzir o limiar tem o efeito oposto: facilita a classificação como branco. Portanto, interpretar `C` exige considerar a polaridade.

### Como escolher a vizinhança?

A janela precisa captar uma referência útil de fundo sem abranger tanta variação de iluminação que deixe de ser local. Em documentos, uma janela muito pequena em relação ao traço pode conter quase exclusivamente a própria letra. Nesse caso, a referência deixa de representar adequadamente o papel ao redor.

Uma janela muito grande pode misturar áreas submetidas a iluminação diferente. O ajuste envolve comparar a largura dos detalhes com a distância ao longo da qual o fundo muda. Na função apresentada, `blockSize` deve ser ímpar e maior que `1`, e a entrada deve ser uma imagem de um canal e 8 bits.

**O adaptativo não é uma versão universalmente superior de Otsu.** Texturas, ruído e áreas uniformes também influenciam a referência local. Em objetos grandes e preenchidos, a parte interior pode não se destacar do seu próprio entorno. O resultado pode enfatizar bordas e perder uma representação sólida do objeto.

### Comparação orientada pela tarefa

| Método | Como obtém a referência | Situação favorável | Limitação a observar |
|---|---|---|---|
| Fixo | Valor definido por você | Aquisição estável e contraste previsível | Sensibilidade a mudanças na intensidade |
| Otsu | Critério global sobre o histograma | Classes tonais suficientemente separadas | Não considera a posição dos pixels |
| Adaptativo | Estatística de cada vizinhança | Variação gradual de iluminação, como em documentos | Depende da escala local e pode destacar textura |

A sintaxe das chamadas e as restrições de parâmetros podem ser consultadas na [documentação oficial de limiarização do OpenCV](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).

---

## 4.8 Morfologia matemática: decisões baseadas na vizinhança

Depois da limiarização, já temos uma seleção de pixels. Entretanto, essa seleção pode conter pequenas imperfeições: um ponto isolado no fundo, uma interrupção em um traço ou um buraco no interior de um objeto. A morfologia permite transformar essas regiões usando uma forma de referência chamada **elemento estruturante**.

Pense em uma pequena sonda que percorre a imagem. Em cada posição, ela observa um conjunto específico de vizinhos. Dependendo da operação, verificamos se todos os pontos exigidos pertencem ao objeto ou se pelo menos um deles pertence. A geometria dessa sonda determina quais vizinhos participam da decisão.

Na limiarização global, a classificação de um pixel depende de sua intensidade e do limiar. Na morfologia binária, o resultado depende da configuração espacial dos pixels na região examinada. Essa diferença explica por que a morfologia consegue eliminar certas estruturas ou conectar regiões, mesmo sem retornar à imagem original.

**A morfologia não identifica o significado das formas.** Se um grão de poeira e um ponto de pontuação têm a mesma forma e escala, uma operação local pode tratá-los da mesma maneira. Para preservar um e remover o outro, talvez seja necessário usar contexto, posição, análise de componentes ou reconhecimento posterior.

Embora a introdução use máscaras binárias, operações morfológicas também se aplicam a tons de cinza. Nesse caso, passam a atuar sobre mínimos e máximos locais de intensidade. Essa extensão será importante em top-hat e black-hat.

---

## 4.9 Elemento estruturante: forma, tamanho e âncora

Um elemento estruturante pode ser representado por uma pequena matriz com posições ativas e inativas. As posições ativas indicam os vizinhos considerados. Uma posição inativa não exige um pixel preto: ela simplesmente não participa do cálculo.

```python
retangular = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eliptico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cruz = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

print("Retangular:\n", retangular)
print("Elíptico:\n", eliptico)
print("Cruz:\n", cruz)
```

O retângulo considera todas as posições da sua área. A elipse utiliza uma aproximação discreta de uma região elíptica. A cruz concentra as posições ativas nos braços horizontal e vertical. Como a imagem é uma grade, uma elipse pequena não possui o contorno contínuo de uma elipse desenhada em papel.

A **âncora** indica a posição do elemento estruturante alinhada ao pixel que está sendo calculado. Usaremos a âncora central padrão e kernels simétricos. Isso simplifica a interpretação de expansão e contração ao redor das regiões.

No OpenCV, o tamanho passado a `getStructuringElement` é `(largura, altura)`. Assim, `(9, 1)` cria um elemento horizontal e `(1, 9)` cria um elemento vertical. Já ao consultar `cinza.shape`, a ordem é `(altura, largura)`. Confundir essas convenções pode mudar a direção do efeito esperado.

### Escolha orientada pela estrutura

Um kernel horizontal pode ser útil para analisar traços horizontais. Um kernel elíptico pode produzir um comportamento mais compatível com objetos arredondados. Um retângulo pode ser apropriado quando todas as posições de uma vizinhança devem participar.

A escolha não é puramente estética. Ela expressa uma hipótese sobre o que deve caber, sobreviver ou se conectar. Sempre observe se a operação está preservando a estrutura relevante para a aplicação.

---

## 4.10 Erosão: exigir que a vizinhança caiba no objeto

Na máscara binária, a erosão mantém um pixel branco somente quando todas as posições ativas do elemento estruturante, alinhado naquela posição, encontram pixels brancos. Basta encontrar um pixel preto em uma posição exigida para que a saída seja preta.

A analogia de uma peça de encaixe ajuda: marque os lugares em que uma pequena peça cabe inteiramente dentro da região branca. Perto da borda, parte da peça fica fora da região e o encaixe falha. Por isso, as bordas recuam e o objeto tende a encolher.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erodida = cv2.erode(binaria, kernel, iterations=1)
```

Considere um quadrado branco sólido de `7 × 7` pixels, cercado por fundo preto e longe das bordas da imagem. Com um kernel retangular de `3 × 3`, uma erosão produz um quadrado de `5 × 5`: perde-se uma camada em cada lado. Sua área passa de `49` para `25` pixels, mostrando que um pequeno recuo na borda pode causar uma mudança proporcional significativa em objetos pequenos.

Um ponto branco isolado não contém espaço para esse kernel e desaparece. Uma ponte de um pixel entre duas regiões também pode desaparecer, separando-as. Uma linha válida com a mesma espessura pode sofrer o mesmo destino.

Em imagens de intensidade, para um elemento estruturante plano, a erosão corresponde ao mínimo entre os valores da vizinhança ativa. Na máscara `0/255`, a presença de qualquer `0` faz o mínimo ser `0`, o que recupera a interpretação binária.

**Atenção aos buracos.** Um buraco preto dentro de uma região branca tende a aumentar com a erosão. A operação reduz o branco tanto na fronteira externa quanto na fronteira interna. Se sua intenção era fechar um buraco, erodir o objeto branco não é a operação apropriada.

---

## 4.11 Dilatação: expandir a presença do primeiro plano

Na dilatação binária, o pixel de saída fica branco se pelo menos uma posição ativa do elemento estruturante encontrar um pixel branco. A exigência é menor que na erosão: não é necessário que toda a vizinhança pertença ao objeto.

Pense em engrossar um traço com uma caneta de ponta mais larga. A área coberta aumenta ao redor da forma existente. Regiões próximas podem se encontrar, e pequenas interrupções podem desaparecer.

```python
dilatada = cv2.dilate(binaria, kernel, iterations=1)
```

No exemplo de um quadrado de `7 × 7` pixels, uma dilatação com retângulo de `3 × 3`, longe da borda da imagem, produz um quadrado de `9 × 9`. A área cresce de `49` para `81` pixels. Isso pode melhorar a continuidade de uma máscara, mas altera diretamente qualquer medida de área feita sobre ela.

Na versão em tons de cinza, a dilatação com elemento plano calcula o máximo da vizinhança ativa. Na máscara binária, basta encontrar um `255` para que o máximo seja `255`.

### Conectar pode ser benefício ou erro

Se um traço foi interrompido por uma falha de aquisição, conectá-lo pode ser desejável. Se duas sementes distintas estão próximas, conectá-las pode transformar dois objetos em um único componente e prejudicar a contagem. A mesma transformação precisa ser julgada à luz da tarefa.

A dilatação não é, em geral, uma operação inversa da erosão. Se a erosão apaga completamente um objeto isolado, não sobra informação que permita à dilatação reconstruí-lo no local original. Essa perda explica o comportamento da abertura.

---

## 4.12 Abertura: remover estruturas que não comportam o kernel

A abertura aplica primeiro erosão e depois dilatação, usando o mesmo elemento estruturante. Na primeira etapa, estruturas que não comportam o kernel desaparecem ou se rompem. Na segunda, as regiões sobreviventes se expandem novamente.

```python
abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
```

A expansão posterior reduz o encolhimento dos objetos que sobreviveram, mas não recupera automaticamente detalhes eliminados. Um ponto isolado removido pela erosão não reaparece apenas porque uma dilatação foi executada em seguida.

Imagine um objeto grande acompanhado de pequenas partículas brancas. Se o kernel cabe no interior do objeto, mas não nas partículas, a abertura pode eliminar essas partículas e manter uma aproximação da forma maior. Essa é a origem da analogia com uma seleção por encaixe.

**Não se trata de um filtro baseado apenas em área.** Uma linha longa e muito fina pode possuir muitos pixels, mas não comportar um kernel quadrado. Um objeto compacto de área menor pode sobreviver. A geometria e a orientação importam tanto quanto a quantidade de pixels.

A abertura também pode remover saliências, romper conexões estreitas e alterar cantos. Se uma conexão fina representa uma parte real do objeto, a operação pode produzir uma segmentação inadequada, mesmo que o fundo pareça mais limpo.

Na formulação usual, a abertura é idempotente: repetir a abertura completa com o mesmo elemento não continua modificando o resultado. Essa propriedade não deve ser confundida com o parâmetro `iterations`, explicado adiante.

---

## 4.13 Fechamento: tratar falhas escuras e pequenas separações

O fechamento aplica primeiro dilatação e depois erosão. A dilatação aproxima as bordas das regiões brancas e pode cobrir falhas escuras. A erosão posterior reduz a expansão externa, mantendo parte das conexões ou preenchimentos obtidos.

```python
fechamento = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
```

Uma analogia é preencher pequenas fissuras em uma superfície e depois retirar o excesso de material. Essa imagem mental ajuda a lembrar a ordem, mas o comportamento real continua determinado pela geometria do elemento estruturante.

Considere uma região branca com um buraco preto muito pequeno. Na dilatação, o branco avança para dentro do buraco. Se a falha é eliminada, a erosão posterior pode preservar o preenchimento. Entretanto, um buraco maior ou incompatível com a escala do kernel pode permanecer.

**Fechamento não significa preencher todos os buracos.** Se a aplicação exige preenchimento de qualquer região interna, independentemente de seu tamanho, pode ser necessário usar um procedimento específico de preenchimento. Aumentar indiscriminadamente o kernel pode unir objetos separados antes de resolver corretamente o problema.

Assim como a abertura, o fechamento completo é idempotente na formulação usual. Ele não representa uma recuperação exata da imagem original; é uma transformação geométrica com efeitos próprios.

| Operação composta | Primeira etapa | Segunda etapa | Efeito frequente no primeiro plano branco |
|---|---|---|---|
| Abertura | Erosão | Dilatação | Remove estruturas brancas que não comportam o kernel |
| Fechamento | Dilatação | Erosão | Fecha certas falhas escuras e separações pequenas |

---

## 4.14 Gradiente morfológico: uma faixa na fronteira

A dilatação faz o objeto avançar; a erosão faz o objeto recuar. Ao subtrair a região erodida da região dilatada, obtemos uma faixa que evidencia a fronteira:

\[
G=\operatorname{dilatação}(B)-\operatorname{erosão}(B).
\]

```python
gradiente = cv2.morphologyEx(binaria, cv2.MORPH_GRADIENT, kernel)
```

Essa faixa inclui a vizinhança da borda e sua espessura depende do elemento estruturante. Não é necessariamente um contorno de um pixel. Um kernel maior costuma produzir uma faixa mais espessa.

O gradiente pode ser útil para destacar limites de regiões, mas não deve ser confundido com afinamento, esqueleto ou uma lista ordenada de pontos de contorno. Se a tarefa exige coordenadas de contornos, será necessário um procedimento adequado, como a extração de contornos da máscara.

Em imagens de cinza, o gradiente com elemento plano corresponde à diferença entre máximo e mínimo locais. Regiões de intensidade quase constante apresentam valores baixos; transições locais fortes apresentam valores maiores. Ruído também pode gerar variações, portanto uma resposta intensa não garante a presença de uma borda relevante.

---

## 4.15 Top-hat e black-hat: destacar diferenças em relação ao entorno

Até aqui, concentramos a discussão em máscaras. Top-hat e black-hat são especialmente úteis quando aplicados à imagem em tons de cinza, antes de uma decisão binária. Eles comparam a imagem com uma transformação morfológica que altera certas estruturas locais.

O **top-hat branco** é a diferença entre a imagem e sua abertura:

\[
\operatorname{TopHat}(I)=I-\operatorname{abertura}(I).
\]

Se a abertura remove um detalhe claro, esse detalhe aparece na diferença. Assim, o top-hat pode realçar pequenos pontos ou traços claros que não são preservados pela abertura com o elemento escolhido.

O **black-hat** é a diferença entre o fechamento e a imagem:

\[
\operatorname{BlackHat}(I)=\operatorname{fechamento}(I)-I.
\]

Se o fechamento preenche um detalhe escuro, a diferença evidencia esse detalhe. Apesar do nome, o detalhe escuro original costuma aparecer como uma resposta clara na imagem resultante.

```python
kernel_realce = cv2.getStructuringElement(
    cv2.MORPH_RECT, (17, 17)
)

tophat = cv2.morphologyEx(cinza, cv2.MORPH_TOPHAT, kernel_realce)
blackhat = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, kernel_realce)
```

### Exemplo: texto em papel com iluminação variável

Suponha que os traços escuros sejam mais estreitos que a escala relevante do kernel e que a iluminação varie de maneira mais lenta. O fechamento pode aproximar uma referência clara ao redor desses traços. A diferença black-hat destaca onde a imagem original era mais escura que essa referência.

Podemos então experimentar uma limiarização sobre o realce:

```python
_, mascara_realce = cv2.threshold(
    blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
```

Usamos aqui a polaridade convencional porque os detalhes realçados no black-hat têm resposta alta. A aparência escura do texto original não determina automaticamente a polaridade de todas as imagens intermediárias.

O tamanho `17 × 17` é apenas ilustrativo. O kernel precisa ser escolhido em relação aos detalhes e à variação do fundo. Se também modificar estruturas que deveriam compor a referência de iluminação, o realce poderá produzir artefatos. Essas operações não garantem uma correção completa da iluminação.

As funções e opções morfológicas estão reunidas na [documentação oficial de transformações morfológicas do OpenCV](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).

---

## 4.16 O tamanho do kernel representa uma escala

Um kernel não é pequeno ou grande de forma absoluta. Um retângulo de `5 × 5` pode ser discreto em uma fotografia de alta resolução e enorme em relação a uma letra com traço de dois pixels. O que importa é sua relação com a estrutura que será tratada.

Antes de escolher o kernel, observe três dimensões: a espessura do menor detalhe válido, o tamanho das imperfeições que deseja modificar e a separação entre objetos que devem permanecer distintos. Esses valores ajudam a antecipar perdas e conexões indesejadas.

Se o ruído tem a mesma escala e geometria de detalhes legítimos, aumentar o kernel não resolve a ambiguidade. Você apenas intensifica uma regra incapaz de distinguir as duas categorias. Nessa situação, será necessário acrescentar outro critério de decisão.

### Resolução e tamanho físico

Suponha que um defeito ocupe quatro pixels de largura. Se a mesma cena, com o mesmo enquadramento, passar a ser amostrada com o dobro de pixels em cada dimensão, o defeito poderá ocupar aproximadamente oito pixels. Manter o kernel antigo altera a escala relativa da operação.

Para preservar aproximadamente o alcance físico de um kernel de lado ímpar, é útil pensar no seu raio. Um kernel de lado `5` tem raio `2`. Dobrando a escala espacial, um raio próximo de `4` corresponde a um lado `9`. Essa relação é uma aproximação; discretização e interpolação podem modificar o resultado.

Em uma aplicação de medição, a relação entre pixels e unidades físicas deve ser calibrada. Além disso, perspectiva e distância da câmera podem fazer o mesmo objeto ocupar tamanhos diferentes em regiões distintas da imagem.

**Registre os parâmetros junto com as condições de aquisição.** Um kernel que funciona em uma câmera e resolução não deve ser transferido automaticamente para outro cenário.

---

## 4.17 Iterações e tratamento das bordas

O parâmetro `iterations` determina quantas vezes a transformação básica é aplicada. Na dilatação, repetições fazem a região avançar progressivamente. Na erosão, fazem a região recuar.

```python
dilatada_tres = cv2.dilate(binaria, kernel, iterations=3)
```

Para um kernel retangular completo de `3 × 3`, três dilatações têm o mesmo alcance geométrico de uma dilatação com retângulo de `7 × 7`, desconsiderando diferenças de tratamento de borda. Essa equivalência depende da composição da forma do kernel; não é uma regra de simplesmente multiplicar o lado por três para qualquer elemento.

Há um detalhe importante nas operações compostas. No OpenCV, `MORPH_OPEN` com `iterations=2` executa duas erosões seguidas de duas dilatações. Isso difere de aplicar duas vezes uma abertura de uma iteração. Assim, não há contradição com a idempotência da abertura completa. A mesma distinção vale para o fechamento.

### O que ocorre na borda da imagem?

Perto do limite do quadro, uma parte da vizinhança fica fora da imagem. O algoritmo precisa de uma regra para lidar com essas posições. Essa escolha pode modificar objetos que tocam a borda.

Os exemplos deste capítulo posicionam os objetos sintéticos longe dos limites. Em aplicações reais, decida se o exterior deve ser considerado fundo, continuidade da imagem ou uma região que não será analisada. Se optar explicitamente por fundo preto, por exemplo, poderá usar `borderType=cv2.BORDER_CONSTANT` e `borderValue=0`, sabendo que isso pode erodir objetos que encostam no quadro.

Para ajustar iterações, comece com uma, inspecione a alteração e verifique o efeito sobre a tarefa. Aumentar até a imagem “parecer limpa” pode apagar detalhes, unir objetos e introduzir erros de medida.

---

## 4.18 Organização do processamento: alternativas e etapas

Um procedimento de processamento, também chamado pipeline, é uma sequência de operações em que a saída de uma etapa alimenta a próxima. A sequência precisa refletir o objetivo e os defeitos observados.

Uma organização inicial possível é: ler a imagem, converter para cinza, suavizar se houver justificativa, escolher um método de limiarização, conferir a polaridade e aplicar uma operação morfológica adequada. Depois, a máscara pode alimentar contagem, extração de contornos ou medição.

**Limiar fixo, Otsu e adaptativo devem ser comparados como alternativas sobre a imagem de intensidade.** Não faz sentido aplicar os três em sequência como se fossem etapas obrigatórias de refinamento. Depois da primeira binarização, boa parte da informação tonal já foi descartada.

Para comparar resultados de forma justa, use a mesma entrada de intensidade e a mesma convenção de primeiro plano. Observe então qual método preserva melhor as regiões relevantes. Se alterar também a suavização, registre essa mudança para não atribuir ao limiarizador um efeito causado pelo filtro.

### A morfologia também exige uma decisão

Não é obrigatório aplicar abertura e fechamento em todo problema. Se a máscara já está adequada, qualquer transformação adicional pode piorá-la. Se o defeito principal são pequenos componentes brancos externos, uma abertura pode ajudar. Se são falhas escuras internas, um fechamento pode ser mais apropriado.

A ordem entre abertura e fechamento pode mudar o resultado. O fechamento pode unir um ponto de ruído a um objeto próximo; depois disso, a abertura talvez não consiga removê-lo como componente isolado. A abertura, por sua vez, pode eliminar uma ligação fina antes que um fechamento posterior tenha condições de preservá-la.

### Critérios para avaliar uma máscara

Se a tarefa é contar, confira se cada objeto corresponde a uma região separada. Se é medir área, compare o quanto a morfologia alterou os limites. Se é ler texto, verifique se os caracteres continuam distinguíveis e se pontos, acentos e separações foram preservados.

Uma máscara mais suave não é necessariamente uma máscara melhor. O critério principal é a fidelidade à informação necessária para a tarefa.

---

## 4.19 Experimento completo: gerar, segmentar e contar

O programa abaixo cria três objetos claros sobre um fundo escuro. Acrescenta pequenas partículas externas e pequenos buracos internos, compara limiarizadores e aplica abertura seguida de fechamento a uma máscara de Otsu.

A geração da cena dentro do próprio código permite executar o experimento sem baixar fotografias. A semente aleatória torna o ruído reproduzível. Como sabemos que a cena foi construída com três objetos principais, temos uma referência para avaliar a contagem.

### Preparação do ambiente

Em um terminal com Python e acesso ao instalador de pacotes, instale as dependências:

```bash
python -m pip install numpy opencv-python matplotlib
```

Salve o código a seguir como `cap04_limiar_morfologia.py` e execute:

```bash
python cap04_limiar_morfologia.py
```

```python
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def contar_componentes(mascara, area_minima=100):
    """Conta regiões brancas conectadas com área >= area_minima."""
    total, rotulos, estatisticas, centroides = (
        cv2.connectedComponentsWithStats(mascara, connectivity=8)
    )

    # O rótulo zero representa o fundo e não entra na contagem.
    areas = estatisticas[1:, cv2.CC_STAT_AREA]
    return int(np.count_nonzero(areas >= area_minima))


def main():
    saida = Path("resultados_cap04")
    saida.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    altura, largura = 280, 420

    # 1. Referência: três objetos principais, sólidos e separados.
    referencia = np.zeros((altura, largura), dtype=np.uint8)
    cv2.circle(referencia, (80, 85), 32, 255, -1)
    cv2.rectangle(referencia, (160, 45), (230, 115), 255, -1)
    cv2.ellipse(referencia, (320, 180), (43, 30), 0, 0, 360, 255, -1)

    # 2. Introduzimos defeitos conhecidos na geometria.
    defeituosa = referencia.copy()
    for centro in [(35, 195), (125, 235), (275, 55), (380, 90)]:
        cv2.circle(defeituosa, centro, 1, 255, -1)

    for centro in [(80, 85), (195, 80), (320, 180)]:
        cv2.circle(defeituosa, centro, 1, 0, -1)

    # 3. Transformamos a máscara em uma cena de intensidades.
    fundo = np.tile(np.linspace(35, 85, largura), (altura, 1))
    cena = np.where(defeituosa > 0, 205.0, fundo)
    ruido = rng.normal(0, 4, size=cena.shape)
    cinza = np.clip(cena + ruido, 0, 255).astype(np.uint8)
    suave = cv2.GaussianBlur(cinza, (3, 3), 0)

    # 4. Comparamos alternativas sobre a mesma imagem de entrada.
    _, fixa = cv2.threshold(suave, 127, 255, cv2.THRESH_BINARY)
    valor_otsu, otsu = cv2.threshold(
        suave, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    adaptativa = cv2.adaptiveThreshold(
        suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )

    # 5. Operações isoladas para comparação.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erodida = cv2.erode(otsu, kernel, iterations=1)
    dilatada = cv2.dilate(otsu, kernel, iterations=1)
    aberta = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    fechada = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    gradiente = cv2.morphologyEx(otsu, cv2.MORPH_GRADIENT, kernel)

    # 6. Sequência escolhida para os defeitos desta cena.
    limpa = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel)

    print(f"Limiar de Otsu: {valor_otsu:.0f}")
    print("Componentes antes, sem filtro de área:",
          contar_componentes(otsu, area_minima=1))
    print("Componentes depois, sem filtro de área:",
          contar_componentes(limpa, area_minima=1))
    print("Objetos depois, com área mínima de 100 pixels:",
          contar_componentes(limpa, area_minima=100))
    print("Referência da cena: 3 objetos principais.")

    # 7. Medida de sobreposição com a referência conhecida.
    previsto = limpa > 0
    esperado = referencia > 0
    intersecao = np.count_nonzero(previsto & esperado)
    uniao = np.count_nonzero(previsto | esperado)
    iou = intersecao / uniao if uniao else 1.0
    print(f"Interseção sobre união (IoU): {iou:.4f}")

    paineis = [
        ("Referência sem defeitos", referencia),
        ("Cena com ruído", cinza),
        ("Suavização", suave),
        ("Limiar fixo", fixa),
        (f"Otsu: {valor_otsu:.0f}", otsu),
        ("Limiar adaptativo", adaptativa),
        ("Erosão de Otsu", erodida),
        ("Dilatação de Otsu", dilatada),
        ("Abertura de Otsu", aberta),
        ("Fechamento de Otsu", fechada),
        ("Abertura + fechamento", limpa),
        ("Gradiente de Otsu", gradiente),
    ]

    plt.rcParams.update({"font.size": 16})
    fig, eixos = plt.subplots(4, 3, figsize=(18, 16))
    for eixo, (titulo, matriz) in zip(eixos.ravel(), paineis):
        eixo.imshow(matriz, cmap="gray", vmin=0, vmax=255)
        eixo.set_title(titulo, fontsize=17)
        eixo.axis("off")

    fig.tight_layout()
    fig.savefig(saida / "painel.png", dpi=160, bbox_inches="tight")

    for nome, matriz in [
        ("cena.png", cinza),
        ("otsu.png", otsu),
        ("mascara_limpa.png", limpa),
    ]:
        if not cv2.imwrite(str(saida / nome), matriz):
            raise OSError(f"Falha ao salvar {nome}")

    print(f"Imagens salvas em: {saida.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
```

### Como interpretar o experimento

A máscara `referencia` representa os objetos antes da introdução de defeitos. A variável `defeituosa` acrescenta alterações geométricas. Em seguida, o programa cria uma imagem com fundo variável e ruído de intensidade. Essas etapas separam três ideias: forma verdadeira, imperfeição da cena e perturbação dos valores dos pixels.

O fundo varia de `35` a `85`, enquanto os objetos ficam próximos de `205`. Essa escolha mantém uma separação tonal favorável aos métodos globais. Assim, o exemplo não foi construído para provar que o adaptativo é melhor. Pelo contrário: ele permite perceber que uma regra local pode selecionar partes do fundo ou deixar de representar adequadamente o interior de objetos grandes.

A comparação entre erosão, dilatação, abertura e fechamento utiliza a mesma máscara `otsu`. Isso permite atribuir as diferenças à operação morfológica. Já `limpa` é o resultado de uma sequência: abertura sobre Otsu e fechamento sobre a abertura.

### Contar regiões não é reconhecer objetos

`connectedComponentsWithStats` agrupa pixels de primeiro plano que estão conectados. Com conectividade `8`, contatos pelas diagonais também conectam pixels. O algoritmo retorna ainda um componente de fundo, identificado pelo rótulo zero, que é excluído da contagem.

O filtro de área mínima é uma decisão adicional. Ele não faz parte da abertura e pode eliminar componentes que sobreviveram à morfologia. Por isso, o programa informa a contagem com e sem esse filtro. Se dois objetos se unirem em uma única região grande, o filtro de área não os separará.

### O que significa a IoU?

A interseção sobre união compara a seleção produzida com a referência:

\[
\operatorname{IoU}=
\frac{\text{pixels selecionados em ambas as máscaras}}
{\text{pixels selecionados em pelo menos uma das máscaras}}.
\]

Uma IoU igual a `1` indica coincidência exata. Valores menores indicam diferenças de inclusão ou exclusão. A métrica permite observar alterações que uma contagem correta não revela: você pode contar três objetos e ainda ter modificado suas áreas.

Não há um valor mínimo universalmente aceitável para toda aplicação. Além disso, uma sobreposição alta pode esconder um erro pequeno em área, mas importante para a tarefa, como uma ponte que une dois objetos. Combine a medida com inspeção visual e verificação da contagem.

**Proposta de exploração.** Modifique apenas um parâmetro por execução. Comece pelo tamanho do kernel, depois altere a intensidade dos objetos e, por fim, aproxime duas formas. Registre o que muda na contagem, na IoU e na preservação das bordas.

---

## 4.20 Erros comuns e raciocínio de diagnóstico

Quando o resultado falha, procure a primeira etapa em que a informação desejada deixa de ser representada. Se a limiarização apagou metade de uma peça por causa de uma sombra, aplicar morfologia indiscriminadamente dificilmente recuperará a geometria correta.

| Sintoma | Hipótese a investigar | Verificação ou ajuste |
|---|---|---|
| A máscara ficou quase toda branca | Limiar baixo na polaridade convencional ou fundo selecionado | Confira polaridade e distribuição tonal |
| O objeto desapareceu na erosão | Kernel incompatível com sua espessura | Reduza alcance e observe o menor detalhe válido |
| O buraco aumentou | Erosão do primeiro plano branco | Compare com fechamento e reveja a intenção |
| Dois objetos passaram a ser um | Dilatação ou fechamento conectou regiões | Observe a distância e reduza o alcance |
| A abertura apagou acentos ou pontos | Detalhes válidos têm escala semelhante ao ruído | Acrescente contexto ou outro critério de seleção |
| Otsu separou sombra de luz | Um corte global não representa as classes desejadas | Avalie correção da aquisição ou método local |
| O adaptativo destacou o fundo | Referência local e parâmetros inadequados | Compare janela, constante e polaridade |
| Cantos e contornos se alteraram | Geometria do kernel afetou a forma | Compare elementos e verifique erro de medida |
| Resultado mudou após redimensionar | A escala em pixels mudou | Ajuste o alcance à nova escala |
| Contagem correta, mas área incorreta | Morfologia mudou as fronteiras | Compare com referência e inspecione a área |
| Objeto na borda sofreu alteração extra | Regra de vizinhança fora do quadro | Inspecione o tratamento de borda |

Um registro simples de experimentos deve informar a imagem utilizada, a polaridade, o método de limiarização, os parâmetros, o kernel e o resultado da tarefa. Compare várias imagens representativas, incluindo situações difíceis. Um ajuste que funciona em uma única cena pode depender de uma coincidência de iluminação ou posicionamento.

---

## 4.21 Revisão guiada com respostas comentadas

### 1. O que a limiarização transforma?

Ela transforma intensidades em classes. No caso binário estudado, cada pixel recebe um de dois valores. O resultado representa uma decisão espacial, mas não fornece automaticamente o nome ou o significado do objeto selecionado.

### 2. Qual é a diferença entre as duas polaridades?

Na regra convencional, valores acima do limiar ficam brancos. Na invertida, valores menores ou iguais ficam brancos. A escolha define qual grupo tonal será tratado como primeiro plano nas operações seguintes.

### 3. Por que olhar o histograma?

Porque ele ajuda a compreender a distribuição das intensidades e a possível separação entre grupos. Entretanto, ele não informa a posição dos pixels nem garante que cada pico corresponda a uma classe semântica.

### 4. O que Otsu otimiza?

Ele escolhe um limiar segundo um critério estatístico de dispersão das classes. Esse critério não mede diretamente se a peça foi corretamente identificada, se uma letra continua legível ou se uma contagem está certa.

### 5. Por que a iluminação irregular pode prejudicar um limiar global?

Porque pixels da mesma classe podem adquirir intensidades diferentes conforme a iluminação. Ao mesmo tempo, pixels de classes distintas podem apresentar intensidades semelhantes. Um único corte não consegue resolver toda sobreposição desse tipo.

### 6. O que o elemento estruturante representa?

Ele define quais posições da vizinhança participam da transformação e em que escala. Sua forma e orientação expressam hipóteses geométricas sobre as estruturas que serão preservadas ou modificadas.

### 7. Por que a erosão reduz o branco?

Porque exige que todas as posições ativas encontrem branco. Perto de uma fronteira, essa condição deixa de ser satisfeita. A região de posições válidas fica menor que a região original em muitos casos usuais.

### 8. Por que a dilatação pode atrapalhar uma contagem?

Porque pode conectar objetos distintos. Um algoritmo de componentes conectados poderá então interpretar os objetos unidos como uma única região, mesmo que visualmente ainda seja possível perceber duas formas.

### 9. A abertura desfaz a erosão?

Não. A dilatação posterior expande o que sobreviveu, mas não possui informação para restaurar toda estrutura que foi eliminada. A remoção de certos detalhes é justamente um dos efeitos pretendidos da abertura.

### 10. Por que o kernel deve acompanhar a escala?

Porque sua ação é definida em pixels. Quando a representação espacial muda, o mesmo kernel passa a ocupar outra proporção do objeto. Preservar um efeito físico aproximado exige considerar essa relação.

---

## Exercícios de fixação com orientação e respostas esperadas

### Parte A — Compreender a decisão por intensidade

#### Exercício 1 — Gradiente e limiar fixo

Crie uma imagem de 100 linhas e 256 colunas em que cada coluna possui a intensidade correspondente ao seu índice:

```python
gradiente = np.tile(np.arange(256, dtype=np.uint8), (100, 1))
```

Aplique os limiares `64`, `128` e `192` com `THRESH_BINARY`. Antes de executar, calcule quantas colunas ficarão brancas em cada caso. Explique por que a coluna igual ao limiar não entra nessa contagem.

**Resposta comentada.** Ficarão brancas, respectivamente, `191`, `127` e `63` colunas. Para o limiar `64`, a faixa branca começa em `65` e termina em `255`. A redução da região branca ilustra o aumento da exigência do corte.

#### Exercício 2 — Otsu em dois grupos tonais

Construa uma imagem com metade dos pixels em `70` e metade em `190`. Aplique Otsu sem ruído. Depois acrescente ruído com semente fixa, limite os valores ao intervalo `0–255` e repita. Registre o limiar e inspecione a máscara, sem exigir que o valor escolhido seja `130`.

**Resposta comentada.** Sem ruído, vários limiares produzem a mesma separação. Com ruído, os grupos se espalham e a escolha depende da nova distribuição. Se a dispersão se tornar grande a ponto de sobrepor as classes, parte dos pixels pode receber a classificação indesejada.

#### Exercício 3 — Iluminação desigual

Crie um fundo em gradiente e adicione traços escuros de espessura conhecida. Compare Otsu invertido com limiar adaptativo invertido. Teste janelas `11`, `31` e `61`, mantendo os demais parâmetros.

**Resposta comentada.** O método local pode preservar traços em regiões com diferentes níveis de iluminação quando a janela fornece uma referência útil do entorno. Não se deve esperar que a menor ou a maior janela seja sempre melhor. Compare continuidade dos traços, inclusão de fundo e preservação dos detalhes.

#### Exercício 4 — Polaridade e interpretação

Aplique as duas polaridades à mesma imagem com o mesmo limiar. Depois dilate cada máscara. Descreva o efeito sobre os objetos escuros da imagem original.

**Resposta comentada.** Quando os objetos escuros estão brancos, a dilatação os expande. Quando estão pretos sobre fundo branco, a expansão do branco pode reduzi-los. A transformação numérica é a mesma; muda a classe associada ao objeto.

### Parte B — Explorar as operações básicas

#### Exercício 5 — Erosão e perda de detalhes

Crie um quadrado branco grande, pequenos pontos brancos e uma linha branca de dois pixels de espessura. Aplique erosão com retângulos `3 × 3`, `5 × 5` e `9 × 9`. Registre o que desaparece primeiro e explique a causa em termos de encaixe.

**Resposta comentada.** A linha fina pode desaparecer mesmo sendo longa. O critério não considera apenas área: o kernel precisa caber na geometria local. O quadrado maior sobrevive por mais tempo, mas perde área a cada aumento de alcance.

#### Exercício 6 — Conexão de regiões

Crie dois retângulos brancos com uma faixa de exatamente três colunas pretas entre eles. Dilate com um retângulo `3 × 3` e conte componentes após uma e duas iterações. Posicione as formas longe das bordas do quadro.

**Resposta comentada.** Na primeira iteração, cada lado avança uma coluna e sobra uma coluna preta. Na segunda, a separação é eliminada e as regiões se conectam. Essa previsão depende da geometria especificada; outras formas de kernel ou disposição podem mudar o resultado.

#### Exercício 7 — Abertura e fechamento isolados

Crie um objeto branco com pequenos buracos pretos e partículas brancas externas. Aplique abertura e fechamento separadamente, sempre sobre a máscara original. Explique qual defeito cada operação tende a tratar.

**Resposta comentada.** A abertura pode remover partículas que não comportam o kernel. O fechamento pode eliminar certas falhas escuras. Avalie também o efeito indesejado: alterar contornos, manter partículas grandes ou unir regiões próximas.

#### Exercício 8 — Trocar a ordem

Na máscara anterior, compare abertura seguida de fechamento com fechamento seguido de abertura. Coloque uma partícula próxima ao objeto para tornar o efeito da ordem mais evidente.

**Resposta comentada.** O fechamento pode incorporar a partícula ao objeto antes que a abertura tente removê-la. Por outro lado, a abertura inicial pode eliminar a partícula enquanto ainda está isolada. Não é obrigatório que as sequências produzam resultados diferentes em toda imagem; a diferença depende das estruturas presentes.

### Parte C — Relacionar forma e escala

#### Exercício 9 — Geometria do kernel

Aplique abertura com retângulo, elipse e cruz de mesmo tamanho a um círculo, uma linha diagonal e um retângulo estreito. Compare quais partes sobrevivem.

**Resposta comentada.** Ter a mesma caixa de dimensões não significa consultar os mesmos vizinhos. As posições ativas diferem e impõem exigências distintas. Relacione o resultado ao formato impresso de cada kernel.

#### Exercício 10 — Mudança de resolução

Dobre a largura e a altura de uma máscara usando interpolação por vizinho mais próximo. Compare o efeito de manter um kernel `5 × 5` e de usar um kernel aproximadamente ajustado à nova escala, como `9 × 9`.

**Resposta comentada.** O kernel mantido passa a ter menor alcance relativo. O ajuste do raio busca preservar aproximadamente o alcance espacial anterior. A equivalência não precisa ser exata devido à representação discreta das bordas.

### Parte D — Aplicar e justificar decisões

#### Exercício 11 — Contagem com análise dos erros

Use o experimento completo e registre contagens antes da morfologia, depois dela e depois do filtro de área. Em seguida, aproxime os objetos até que duas regiões se unam. Explique por que o filtro de área não resolve essa união.

**Resposta comentada.** O filtro elimina componentes com área fora do critério adotado, mas não separa um componente em objetos individuais. Uma contagem confiável depende de uma segmentação que mantenha separadas as entidades que serão contadas.

#### Exercício 12 — Realce com top-hat

Crie traços claros sobre um fundo que varia lentamente. Aplique top-hat com três tamanhos de kernel e observe o que acontece quando o elemento é pequeno demais para remover os traços na abertura.

**Resposta comentada.** Se a abertura preserva os traços, a diferença top-hat tende a ser pequena neles. Para destacá-los, a operação precisa alterar essas estruturas de maneira diferente do fundo. Um kernel excessivo também pode responder a variações indesejadas do entorno.

#### Exercício 13 — Realce com black-hat

Crie ou fotografe texto escuro sobre fundo claro variável. Aplique black-hat e depois uma limiarização convencional. Explique por que o texto original era escuro, mas o realce pode ser selecionado como claro.

**Resposta comentada.** O resultado expressa uma diferença positiva entre o fechamento e a imagem. Onde havia uma depressão escura preenchida pelo fechamento, aparece uma resposta alta. A polaridade deve acompanhar essa nova representação.

#### Exercício 14 — Limite da interpretação geométrica

Uma imagem contém um ponto de pontuação e um grão de poeira com aparência local idêntica. Explique por que uma abertura com o mesmo kernel não consegue preservar um apenas por ser “correto” e remover o outro por ser “ruído”. Proponha informações adicionais.

**Resposta comentada.** A operação não recebe esses significados. Ela responde à configuração local. Posição em relação às letras, alinhamento com a linha de texto, reconhecimento de caracteres ou informação de outra captura podem fornecer critérios adicionais.

---

## Encerramento do capítulo

Limiarizar é estabelecer uma regra que converte intensidades em uma seleção espacial. Escolher essa regra exige compreender o contraste, a iluminação e a distribuição dos valores. O limiar fixo oferece controle direto; Otsu automatiza uma escolha global; o adaptativo usa referências locais. Cada método depende de condições que precisam ser verificadas na imagem e na tarefa.

A morfologia modifica a geometria dessa seleção. Erosão e dilatação alteram a presença do primeiro plano conforme uma vizinhança definida. Abertura e fechamento combinam essas operações para tratar determinadas estruturas, enquanto gradiente, top-hat e black-hat evidenciam diferenças espaciais ou de intensidade.

O aprendizado central é justificar cada transformação. Antes de aplicar uma operação, identifique o que deseja preservar, qual defeito pretende modificar e em que escala ambos aparecem. Depois, examine se o resultado continua adequado para contar, medir ou reconhecer. Essa ligação entre parâmetro, efeito e objetivo transforma uma sequência de funções em um procedimento de análise compreensível.

## Referências e leituras de apoio

As referências bibliográficas do material de origem foram mantidas abaixo. Os links da documentação oficial complementam a consulta às funções utilizadas; as situações didáticas e o experimento sintético foram desenvolvidos para este capítulo.

GONZALEZ, Rafael C.; WOODS, Richard E. *Processamento digital de imagens*. 3. ed. São Paulo: Pearson, 2010.

OTSU, Nobuyuki. A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, v. 9, n. 1, p. 62–66, 1979.

SZELISKI, Richard. *Computer vision: algorithms and applications*. 2. ed. Cham: Springer, 2022.

OPENCV. *Image thresholding*. OpenCV documentation, [s. d.]. Disponível em: [documentação de limiarização](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html). Acesso em: 15 set. 2026.

OPENCV. *Morphological transformations*. OpenCV documentation, [s. d.]. Disponível em: [documentação de morfologia](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html). Acesso em: 15 set. 2026.
