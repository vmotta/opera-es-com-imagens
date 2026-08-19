"""Capítulo 10: preparação de blob e interpretação de saída SSD simulada."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
]


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.full((600, 800, 3), 155, dtype=np.uint8)
    cv2.rectangle(imagem, (345, 190), (455, 515), (50, 90, 210), -1)
    cv2.circle(imagem, (400, 145), 48, (90, 150, 220), -1)

    blob = cv2.dnn.blobFromImage(
        imagem,
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
        crop=False,
    )
    print(f"Blob NCHW: {blob.shape}; intervalo [{blob.min():.3f}, {blob.max():.3f}]")

    # Estrutura do SSD: [imagem, canal, detecção, (id, confiança, x1, y1, x2, y2)].
    deteccoes = np.array(
        [[[[0, 15, 0.985, 345 / 800, 95 / 600, 455 / 800, 515 / 600]]]],
        dtype=np.float32,
    )
    resultado = imagem.copy()
    h, w = resultado.shape[:2]
    for indice in range(deteccoes.shape[2]):
        confianca = float(deteccoes[0, 0, indice, 2])
        if confianca < 0.5:
            continue
        classe = int(deteccoes[0, 0, indice, 1])
        caixa = deteccoes[0, 0, indice, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = caixa.astype(int)
        cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            resultado,
            f"{CLASSES[classe]}: {confianca:.1%}",
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 100, 0),
            2,
        )

    salvar(output_dir / "01_entrada_sintetica.png", imagem)
    salvar(output_dir / "02_saida_ssd_simulada.png", resultado)


def main() -> None:
    parser = criar_parser("cap10", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
