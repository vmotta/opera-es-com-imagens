"""Capítulo 17: espaço latente, diversidade e arquiteturas GAN opcionais."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def gerador_didatico(z: np.ndarray, tamanho: int = 192) -> np.ndarray:
    """Mapeamento visual contínuo para estudar espaço latente; NÃO é uma GAN treinada."""
    tela = np.full((tamanho, tamanho, 3), 245, dtype=np.uint8)
    centro = (
        int(tamanho / 2 + np.tanh(z[0]) * 38),
        int(tamanho / 2 + np.tanh(z[1]) * 38),
    )
    raio = int(24 + (np.tanh(z[2]) + 1) * 26)
    cor = tuple(int(np.clip(35 + (np.tanh(valor) + 1) * 95, 0, 255)) for valor in z[3:6])
    angulo = float(np.tanh(z[6] if len(z) > 6 else 0.0) * 70)
    eixos = (raio, max(12, int(raio * (0.55 + 0.25 * (np.tanh(z[7] if len(z) > 7 else 0.0) + 1)))))
    cv2.ellipse(tela, centro, eixos, angulo, 0, 360, cor, -1, cv2.LINE_AA)
    return tela


def diversidade_pixelar(imagens: list[np.ndarray]) -> float:
    """Métrica didática simples: variância média entre imagens; não substitui FID."""
    pilha = np.stack([img.astype(np.float32) / 255.0 for img in imagens], axis=0)
    return float(np.mean(np.var(pilha, axis=0)))


def construir_dcgan_opcional():
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print('[INFO] TensorFlow ausente. Instale com: python -m pip install -e ".[deep]"')
        return None, None

    discriminador = keras.Sequential(
        [
            layers.Input((64, 64, 3)),
            layers.Conv2D(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminador",
    )
    gerador_transposto = keras.Sequential(
        [
            layers.Input((100,)),
            layers.Dense(8 * 8 * 128),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh"),
        ],
        name="gerador_conv_transposta",
    )
    gerador_resize = keras.Sequential(
        [
            layers.Input((100,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.UpSampling2D(2),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.UpSampling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.UpSampling2D(2),
            layers.Conv2D(3, 3, padding="same", activation="tanh"),
        ],
        name="gerador_resize_conv",
    )

    print("\n=== DISCRIMINADOR ===")
    discriminador.summary()
    print("\n=== GERADOR COM CONV2DTRANSPOSE ===")
    gerador_transposto.summary()
    print("\n=== GERADOR COM UPSAMPLING + CONV ===")
    gerador_resize.summary()
    print(
        f"Parâmetros: D={discriminador.count_params():,}; "
        f"G transpose={gerador_transposto.count_params():,}; "
        f"G resize+conv={gerador_resize.count_params():,}"
    )
    return gerador_transposto, discriminador


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    rng = np.random.default_rng(17)

    # --------------------------------------------------------------------------
    # 1. INTERPOLAÇÃO: mudanças suaves entre dois vetores.
    # --------------------------------------------------------------------------
    inicio = np.array([-1.5, -0.8, -1.0, -1.2, 0.2, 1.0, -0.6, -0.4])
    fim = np.array([1.2, 1.0, 1.5, 0.8, -0.8, -1.0, 1.0, 0.9])
    interpolacoes = []
    imagens_interpoladas = []
    for indice, alfa in enumerate(np.linspace(0, 1, 10)):
        z = (1 - alfa) * inicio + alfa * fim
        img = gerador_didatico(z)
        imagens_interpoladas.append(img)
        interpolacoes.append(rotular(img, f"alpha={alfa:.2f}"))
    painel_interp = mosaico(interpolacoes, colunas=5, largura=220)
    salvar(output_dir / "01_interpolacao_latente_didatica.png", painel_interp)

    # --------------------------------------------------------------------------
    # 2. AMOSTRAGEM ALEATÓRIA: evita escolher somente exemplos "bonitos".
    # --------------------------------------------------------------------------
    amostras_aleatorias = []
    imgs_aleatorias = []
    for i in range(16):
        z = rng.normal(size=8)
        img = gerador_didatico(z)
        imgs_aleatorias.append(img)
        amostras_aleatorias.append(rotular(img, f"z aleatorio {i+1}"))
    painel_random = mosaico(amostras_aleatorias, colunas=4, largura=190)
    salvar(output_dir / "02_amostras_aleatorias.png", painel_random)

    # --------------------------------------------------------------------------
    # 3. MODE COLLAPSE DIDÁTICO: ignora z e repete quase a mesma saída.
    # --------------------------------------------------------------------------
    base_colapso = np.array([0.15, -0.10, 0.25, 0.4, -0.3, 0.2, 0.1, 0.0])
    imgs_colapso = []
    colapso_rotulado = []
    for i in range(16):
        z = base_colapso + rng.normal(0, 0.01, size=8)
        img = gerador_didatico(z)
        imgs_colapso.append(img)
        colapso_rotulado.append(rotular(img, f"saida {i+1}"))
    painel_colapso = mosaico(colapso_rotulado, colunas=4, largura=190)
    salvar(output_dir / "03_mode_collapse_didatico.png", painel_colapso)

    div_random = diversidade_pixelar(imgs_aleatorias)
    div_colapso = diversidade_pixelar(imgs_colapso)
    print("A função visual usada aqui NÃO é uma GAN treinada; serve apenas como analogia de z -> saída.")
    print(f"Diversidade pixelar didática — amostras variadas={div_random:.6f}; colapso={div_colapso:.6f}")
    print("Essa variância NÃO substitui FID nem métricas de modelos generativos reais.")

    # Quadro conceitual do treinamento alternado.
    quadro = np.full((500, 900, 3), 248, dtype=np.uint8)
    etapas = [
        "TREINAMENTO ADVERSARIAL — CICLO DIDATICO",
        "1. amostrar imagens reais x",
        "2. amostrar vetor latente z",
        "3. gerar falsas G(z)",
        "4. atualizar D para separar reais e falsas",
        "5. amostrar novo z",
        "6. manter pesos de D fixos nesta etapa",
        "7. atualizar G atraves do gradiente que passa por D",
        "8. repetir e avaliar fidelidade + diversidade",
    ]
    for i, texto in enumerate(etapas):
        cv2.putText(quadro, texto, (35, 55 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, 0.68 if i else 0.82, (30, 30, 30), 2, cv2.LINE_AA)
    salvar(output_dir / "04_ciclo_adversarial.png", quadro)

    construir_dcgan_opcional()


def main() -> None:
    parser = criar_parser("cap17", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
