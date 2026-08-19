# 17. Redes Adversárias Generativas

## Dois objetivos em competição

Uma GAN combina:

- **Gerador `G(z)`:** transforma ruído latente em amostras;
- **Discriminador `D(x)`:** estima se uma amostra parece real.

O gerador funciona como falsificador; o discriminador como perito. Quando o perito identifica um defeito, o gradiente informa ao falsificador como melhorar. Diferentemente de uma competição humana, ambos são otimizadores diferenciáveis.

Uma forma clássica do jogo minimax é:

\[
\min_G\max_D \; \mathbb{E}_{x\sim p_{dados}}[\log D(x)]
+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
\]

Na prática, funções de perda e técnicas modernas variam para melhorar gradientes e estabilidade.

## Espaço latente

`z` é um vetor amostrado de uma distribuição simples. O gerador aprende a dobrar esse espaço em uma distribuição visual. Interpolar entre dois vetores costuma produzir transições graduais quando a representação foi bem aprendida.

O [código do capítulo 17](https://github.com/vmotta/opera-es-com-imagens/blob/main/exemplos/cap17_gan.py) inclui uma analogia explícita: uma função programada transforma `z` em posição, tamanho e cor. Ela **não é uma GAN treinada**, mas torna visível a ideia de continuidade latente.

![Interpolação didática entre dois vetores latentes](../assets/resultados/cap17/01_interpolacao_latente_didatica.png)

## DCGAN: reduzir e ampliar

O discriminador usa convoluções com stride para reduzir `64 → 32 → 16` e emitir decisão. O gerador começa com vetor denso, reorganiza para uma grade pequena e usa convoluções transpostas para ampliar `8 → 16 → 32 → 64`.

Convolução transposta não é inversa perfeita da convolução; é uma operação aprendível de aumento. Certas combinações kernel/stride geram artefatos quadriculados. Upsampling seguido de convolução é uma alternativa.

```bash
python -m exemplos.cap17_gan
# para também montar as arquiteturas Keras
python -m pip install -e ".[deep]"
```

## Treinamento alternado

Um ciclo simplificado:

1. amostra imagens reais;
2. gera imagens falsas;
3. atualiza `D` para separar reais e falsas;
4. congela `D` para a etapa do gerador;
5. atualiza `G` para aumentar a pontuação das falsas;
6. repete e avalia amostras e métricas.

Equilíbrio não significa necessariamente loss `0,5` nem imagens boas. O discriminador pode dominar; o gerador pode sofrer **mode collapse** e produzir pouca variedade.

## Avaliação e uso responsável

FID compara estatísticas de features entre conjuntos, mas depende de amostragem e domínio. Inspeção visual isolada seleciona casos favoráveis. Avalie fidelidade e diversidade.

Dados e saídas generativas podem reproduzir vieses, pessoas ou conteúdo protegido. Documente dataset, consentimento/licença, limitações e origem sintética; não apresente geração como registro real.

## Exercícios

1. Desenhe o fluxo de gradiente quando o gerador é atualizado.
2. Explique por que um gerador que produz uma única imagem perfeita falha em diversidade.
3. Compare `Conv2DTranspose` com `UpSampling2D + Conv2D` quanto a artefatos.
