"""Capítulo 10: blobs, pré-processamento e interpretação segura de saída SSD simulada."""

from __future__ import annotations

import time

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
]


def limitar_caixa(x1: int, y1: int, x2: int, y2: int, largura: int, altura: int):
    x1 = max(0, min(x1, largura - 1))
    y1 = max(0, min(y1, altura - 1))
    x2 = max(0, min(x2, largura - 1))
    y2 = max(0, min(y2, altura - 1))
    return x1, y1, x2, y2


def interpretar_ssd(imagem: np.ndarray, deteccoes: np.ndarray, limiar: float) -> tuple[np.ndarray, int]:
    """Interpreta uma saída SSD didática e rejeita caixas inválidas."""
    resultado = imagem.copy()
    h, w = resultado.shape[:2]
    mantidas = 0
    for indice in range(deteccoes.shape[2]):
        registro = deteccoes[0, 0, indice]
        classe = int(registro[1])
        confianca = float(registro[2])
        if confianca < limiar:
            continue

        caixa = registro[3:7] * np.array([w, h, w, h], dtype=np.float32)
        x1, y1, x2, y2 = caixa.astype(int)
        x1, y1, x2, y2 = limitar_caixa(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            print(f"Detecção {indice} rejeitada: caixa sem área após clipping")
            continue

        mantidas += 1
        rotulo = CLASSES[classe] if 0 <= classe < len(CLASSES) else f"classe_{classe}"
        cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            resultado,
            f"{rotulo}: {confianca:.2f}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 100, 0),
            2,
        )
    return resultado, mantidas


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    imagem = np.full((600, 800, 3), 155, dtype=np.uint8)
    cv2.rectangle(imagem, (345, 190), (455, 515), (50, 90, 210), -1)
    cv2.circle(imagem, (400, 145), 48, (90, 150, 220), -1)
    cv2.rectangle(imagem, (60, 390), (220, 520), (210, 80, 60), -1)

    # --------------------------------------------------------------------------
    # 1. MESMA IMAGEM, TRÊS CONTRATOS NUMÉRICOS PARA OBSERVAR DIFERENÇAS.
    # --------------------------------------------------------------------------
    inicio = time.perf_counter()
    blob_mobilenet = cv2.dnn.blobFromImage(
        imagem,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
        crop=False,
    )
    tempo_blob = time.perf_counter() - inicio

    blob_01 = cv2.dnn.blobFromImage(
        imagem,
        scalefactor=1 / 255.0,
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False,
    )
    blob_rgb = cv2.dnn.blobFromImage(
        imagem,
        scalefactor=1 / 255.0,
        size=(224, 224),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )

    for nome, blob in [
        ("MobileNet-SSD", blob_mobilenet),
        ("0..1 BGR", blob_01),
        ("0..1 RGB", blob_rgb),
    ]:
        print(
            f"{nome}: shape={blob.shape}, dtype={blob.dtype}, "
            f"min={blob.min():.3f}, max={blob.max():.3f}, média={blob.mean():.3f}"
        )
    print(f"Tempo de blobFromImage: {tempo_blob * 1000:.3f} ms")
    print("Primeiro pixel/canal BGR blob:", blob_01[0, :, 0, 0].tolist())
    print("Primeiro pixel/canal swapRB blob:", blob_rgb[0, :, 0, 0].tolist())

    # --------------------------------------------------------------------------
    # 2. SAÍDA SSD SIMULADA COM CASOS FORTES, FRACOS E CAIXA FORA DA IMAGEM.
    # Formato: [image_id, class_id, confidence, x1, y1, x2, y2].
    # --------------------------------------------------------------------------
    registros = [
        [0, 15, 0.985, 345 / 800, 95 / 600, 455 / 800, 515 / 600],
        [0, 7, 0.72, 55 / 800, 385 / 600, 230 / 800, 530 / 600],
        [0, 15, 0.43, 0.40, 0.20, 0.56, 0.87],
        [0, 7, 0.83, -0.06, 0.72, 0.22, 1.08],
        [0, 7, 0.91, 0.70, 0.60, 0.68, 0.62],  # inválida x2 < x1
    ]
    deteccoes = np.array(registros, dtype=np.float32).reshape(1, 1, len(registros), 7)

    resultado_040, n040 = interpretar_ssd(imagem, deteccoes, 0.40)
    resultado_070, n070 = interpretar_ssd(imagem, deteccoes, 0.70)
    resultado_090, n090 = interpretar_ssd(imagem, deteccoes, 0.90)
    print(f"Detecções válidas: limiar .40={n040}; .70={n070}; .90={n090}")

    # --------------------------------------------------------------------------
    # 3. VISUALIZA O EFEITO DA DEFORMAÇÃO AO FORÇAR 800x600 PARA 300x300.
    # --------------------------------------------------------------------------
    deformada = cv2.resize(imagem, (300, 300), interpolation=cv2.INTER_LINEAR)
    preservada = cv2.resize(imagem, (300, 225), interpolation=cv2.INTER_AREA)

    salvar(output_dir / "01_entrada_sintetica.png", imagem)
    salvar(output_dir / "02_saida_ssd_simulada.png", resultado_070)
    salvar(output_dir / "03_limiar_040.png", resultado_040)
    salvar(output_dir / "04_limiar_070.png", resultado_070)
    salvar(output_dir / "05_limiar_090.png", resultado_090)
    salvar(output_dir / "06_resize_deformado_300x300.png", deformada)
    salvar(output_dir / "07_resize_proporcional_300x225.png", preservada)

    painel = mosaico(
        [
            rotular(imagem, "Entrada 800x600"),
            rotular(resultado_040, "SSD limiar 0.40"),
            rotular(resultado_070, "SSD limiar 0.70"),
            rotular(resultado_090, "SSD limiar 0.90"),
            rotular(deformada, "Resize 300x300: deformado"),
            rotular(preservada, "Resize proporcional"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap10", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
