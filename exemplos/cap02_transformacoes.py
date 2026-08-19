"""Capítulo 2: escala, interpolação, translação, rotação, afim e perspectiva."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.full((320, 320, 3), 225, dtype=np.uint8)
    cv2.rectangle(imagem, (45, 45), (275, 275), (0, 105, 205), -1)
    cv2.circle(imagem, (160, 160), 62, (210, 55, 40), -1)
    cv2.putText(imagem, "OpenCV", (85, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    altura, largura = imagem.shape[:2]

    ampliada = cv2.resize(imagem, (640, 640), interpolation=cv2.INTER_CUBIC)
    reduzida = cv2.resize(imagem, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    matriz_translacao = np.float32([[1, 0, 52], [0, 1, 34]])
    transladada = cv2.warpAffine(imagem, matriz_translacao, (largura, altura))

    matriz_rotacao = cv2.getRotationMatrix2D((largura / 2, altura / 2), 35, 1.0)
    rotacionada = cv2.warpAffine(
        imagem,
        matriz_rotacao,
        (largura, altura),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(225, 225, 225),
    )

    origem_afim = np.float32([[45, 45], [275, 45], [45, 275]])
    destino_afim = np.float32([[20, 85], [280, 40], [85, 290]])
    afim = cv2.warpAffine(
        imagem,
        cv2.getAffineTransform(origem_afim, destino_afim),
        (largura, altura),
    )

    origem_perspectiva = np.float32([[45, 45], [275, 45], [45, 275], [275, 275]])
    destino_perspectiva = np.float32([[15, 35], [305, 5], [60, 305], [270, 275]])
    perspectiva = cv2.warpPerspective(
        imagem,
        cv2.getPerspectiveTransform(origem_perspectiva, destino_perspectiva),
        (largura, altura),
    )

    resultados = {
        "01_original.png": imagem,
        "02_ampliada_cubica.png": ampliada,
        "03_reduzida_area.png": reduzida,
        "04_transladada.png": transladada,
        "05_rotacionada.png": rotacionada,
        "06_afim.png": afim,
        "07_perspectiva.png": perspectiva,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)

    painel = mosaico(
        [
            rotular(imagem, "Original"),
            rotular(reduzida, "Reducao INTER_AREA"),
            rotular(transladada, "Translacao"),
            rotular(rotacionada, "Rotacao 35 graus"),
            rotular(afim, "Transformacao afim"),
            rotular(perspectiva, "Perspectiva"),
        ]
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap02", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
