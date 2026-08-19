"""Capítulo 17: espaço latente e arquiteturas de uma DCGAN."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def gerador_didatico(z: np.ndarray, tamanho: int = 192) -> np.ndarray:
    """Visualiza uma transformação latente; não é uma GAN treinada."""
    tela = np.full((tamanho, tamanho, 3), 245, dtype=np.uint8)
    centro = (
        int(tamanho / 2 + np.tanh(z[0]) * 35),
        int(tamanho / 2 + np.tanh(z[1]) * 35),
    )
    raio = int(28 + (np.tanh(z[2]) + 1) * 28)
    cor = tuple(int(45 + (np.tanh(valor) + 1) * 90) for valor in z[3:6])
    cv2.circle(tela, centro, raio, cor, -1, cv2.LINE_AA)
    return tela


def construir_dcgan_opcional():
    try:
        from tensorflow.keras.layers import (
            Conv2D,
            Conv2DTranspose,
            Dense,
            Dropout,
            Flatten,
            LeakyReLU,
            Reshape,
        )
        from tensorflow.keras.models import Sequential
    except ImportError:
        print('[INFO] TensorFlow ausente. Instale com: python -m pip install -e ".[deep]"')
        return None, None

    discriminador = Sequential(
        [
            Conv2D(64, 4, strides=2, padding="same", input_shape=(64, 64, 3)),
            LeakyReLU(negative_slope=0.2),
            Dropout(0.3),
            Conv2D(128, 4, strides=2, padding="same"),
            LeakyReLU(negative_slope=0.2),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminador",
    )
    gerador = Sequential(
        [
            Dense(8 * 8 * 128, input_shape=(100,)),
            LeakyReLU(negative_slope=0.2),
            Reshape((8, 8, 128)),
            Conv2DTranspose(128, 4, strides=2, padding="same"),
            LeakyReLU(negative_slope=0.2),
            Conv2DTranspose(64, 4, strides=2, padding="same"),
            LeakyReLU(negative_slope=0.2),
            Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh"),
        ],
        name="gerador",
    )
    discriminador.summary()
    gerador.summary()
    return gerador, discriminador


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    inicio = np.array([-1.5, -0.8, -1.0, -1.2, 0.2, 1.0])
    fim = np.array([1.2, 1.0, 1.5, 0.8, -0.8, -1.0])
    amostras = []
    for indice, alfa in enumerate(np.linspace(0, 1, 8)):
        z = (1 - alfa) * inicio + alfa * fim
        amostras.append(rotular(gerador_didatico(z), f"z: passo {indice + 1}"))
    salvar(output_dir / "01_interpolacao_latente_didatica.png", mosaico(amostras, colunas=4, largura=220))
    print("A figura é uma analogia programada do espaço latente, não a saída de uma GAN treinada.")
    construir_dcgan_opcional()


def main() -> None:
    parser = criar_parser("cap17", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
