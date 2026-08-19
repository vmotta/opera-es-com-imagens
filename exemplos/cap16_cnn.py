"""Capítulo 16: convolução, ReLU, pooling, shapes, parâmetros e CNN Keras opcional."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def dimensao_convolucao(n: int, kernel: int, padding: int = 0, stride: int = 1) -> int:
    return (n + 2 * padding - kernel) // stride + 1


def parametros_conv(kernel_h: int, kernel_w: int, canais_entrada: int, filtros: int, bias=True) -> int:
    por_filtro = kernel_h * kernel_w * canais_entrada + (1 if bias else 0)
    return por_filtro * filtros


def max_pooling_2x2(matriz: np.ndarray) -> np.ndarray:
    altura = matriz.shape[0] - matriz.shape[0] % 2
    largura = matriz.shape[1] - matriz.shape[1] % 2
    blocos = matriz[:altura, :largura].reshape(altura // 2, 2, largura // 2, 2)
    return blocos.max(axis=(1, 3))


def construir_cnn_opcional():
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print('[INFO] TensorFlow ausente. Instale com: python -m pip install -e ".[deep]"')
        return None

    modelo_gap = keras.Sequential(
        [
            layers.Input((64, 64, 3)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(3, activation="softmax"),
        ],
        name="cnn_com_gap",
    )

    modelo_flat = keras.Sequential(
        [
            layers.Input((64, 64, 3)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
        name="cnn_com_flatten",
    )

    print("\n=== CNN COM GLOBAL AVERAGE POOLING ===")
    modelo_gap.summary()
    print("\n=== CNN COM FLATTEN ===")
    modelo_flat.summary()
    print(f"Parâmetros GAP={modelo_gap.count_params():,}; Flatten={modelo_flat.count_params():,}")
    return modelo_gap


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    imagem = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(imagem, (45, 55), (215, 205), 200, -1)
    cv2.circle(imagem, (130, 130), 52, 70, -1)
    cv2.line(imagem, (20, 230), (235, 230), 255, 4)

    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_horizontal = kernel_vertical.T
    mapa_vertical = cv2.filter2D(imagem.astype(np.float32), cv2.CV_32F, kernel_vertical)
    mapa_horizontal = cv2.filter2D(imagem.astype(np.float32), cv2.CV_32F, kernel_horizontal)
    relu_vertical = np.maximum(mapa_vertical, 0)
    pooling = max_pooling_2x2(relu_vertical)

    def visualizar(matriz):
        return cv2.normalize(matriz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --------------------------------------------------------------------------
    # 1. CÁLCULOS MANUAIS DE SHAPE E PARÂMETROS.
    # --------------------------------------------------------------------------
    print("\n=== CÁLCULO DE DIMENSÕES ===")
    for n, k, p, s in [(32, 3, 0, 1), (32, 3, 1, 1), (64, 5, 0, 2), (64, 3, 1, 2)]:
        print(f"entrada={n}, K={k}, P={p}, S={s} -> saída={dimensao_convolucao(n, k, p, s)}")

    print("\n=== CÁLCULO DE PARÂMETROS ===")
    exemplos = [
        (3, 3, 3, 32),
        (5, 5, 3, 16),
        (3, 3, 32, 64),
    ]
    for kh, kw, cin, fout in exemplos:
        total = parametros_conv(kh, kw, cin, fout)
        print(f"Conv {kh}x{kw}, Cin={cin}, filtros={fout} -> {total:,} parâmetros")

    # --------------------------------------------------------------------------
    # 2. FLATTEN VERSUS GAP EM UM TENSOR HIPOTÉTICO 16x16x64.
    # --------------------------------------------------------------------------
    flatten_dim = 16 * 16 * 64
    gap_dim = 64
    dense_flat_params = (flatten_dim + 1) * 128
    dense_gap_params = (gap_dim + 1) * 128
    print(
        f"16x16x64 -> Flatten={flatten_dim} valores; GAP={gap_dim}; "
        f"Dense128 após Flatten={dense_flat_params:,} params; após GAP={dense_gap_params:,} params"
    )

    painel = mosaico(
        [
            rotular(imagem, "Entrada"),
            rotular(visualizar(mapa_vertical), "Feature vertical"),
            rotular(visualizar(mapa_horizontal), "Feature horizontal"),
            rotular(visualizar(relu_vertical), "Depois da ReLU"),
            rotular(visualizar(pooling), "Max pooling 2x2"),
        ],
        colunas=3,
    )
    salvar(output_dir / "01_feature_maps.png", painel)

    # Quadro textual grande para consolidar shapes/parâmetros visualmente.
    quadro = np.full((460, 920, 3), 248, dtype=np.uint8)
    linhas = [
        "CNN: SHAPES E PARAMETROS",
        "Conv 3x3, Cin=3, Cout=32: (3*3*3+1)*32 = 896",
        "64x64x3 --Conv same 32--> 64x64x32",
        "--Pool2--> 32x32x32 --Conv64--> 32x32x64",
        "--Pool2--> 16x16x64",
        f"Flatten: {flatten_dim} valores | GAP: {gap_dim} valores",
        f"Dense128: Flatten={dense_flat_params:,} params | GAP={dense_gap_params:,}",
        "Softmax: classes exclusivas | Sigmoid: multirrotulo",
    ]
    for i, texto in enumerate(linhas):
        cv2.putText(
            quadro,
            texto,
            (30, 55 + i * 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if i else 0.85,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    salvar(output_dir / "02_shapes_parametros.png", quadro)

    construir_cnn_opcional()


def main() -> None:
    parser = criar_parser("cap16", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
