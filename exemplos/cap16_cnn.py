"""Capítulo 16: feature maps, ReLU, pooling e arquitetura CNN opcional."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def max_pooling_2x2(matriz: np.ndarray) -> np.ndarray:
    altura = matriz.shape[0] - matriz.shape[0] % 2
    largura = matriz.shape[1] - matriz.shape[1] % 2
    blocos = matriz[:altura, :largura].reshape(altura // 2, 2, largura // 2, 2)
    return blocos.max(axis=(1, 3))


def construir_cnn_opcional():
    try:
        from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
        from tensorflow.keras.models import Sequential
    except ImportError:
        print('[INFO] TensorFlow ausente. Instale com: python -m pip install -e ".[deep]"')
        return None

    modelo = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(3, activation="softmax"),
        ],
        name="cnn_didatica",
    )
    modelo.summary()
    return modelo


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(imagem, (45, 55), (215, 205), 200, -1)
    cv2.circle(imagem, (130, 130), 52, 70, -1)

    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_horizontal = kernel_vertical.T
    mapa_vertical = cv2.filter2D(imagem.astype(np.float32), cv2.CV_32F, kernel_vertical)
    mapa_horizontal = cv2.filter2D(imagem.astype(np.float32), cv2.CV_32F, kernel_horizontal)
    relu_vertical = np.maximum(mapa_vertical, 0)
    pooling = max_pooling_2x2(relu_vertical)

    def visualizar(matriz):
        return cv2.normalize(matriz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    painel = mosaico(
        [
            rotular(imagem, "Entrada"),
            rotular(visualizar(mapa_vertical), "Feature vertical"),
            rotular(visualizar(mapa_horizontal), "Feature horizontal"),
            rotular(visualizar(relu_vertical), "Depois da ReLU"),
            rotular(visualizar(pooling), "Max pooling 2x2"),
        ]
    )
    salvar(output_dir / "01_feature_maps.png", painel)
    print(f"Dimensões: entrada={imagem.shape}, após pooling={pooling.shape}")
    construir_cnn_opcional()


def main() -> None:
    parser = criar_parser("cap16", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
