"""Capítulo 4: Otsu, erosão, dilatação, abertura e fechamento."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    gradiente = np.linspace(20, 50, 400, dtype=np.uint8)
    imagem = np.tile(gradiente, (400, 1))
    cv2.rectangle(imagem, (95, 95), (305, 305), 205, -1)
    for centro, raio in [((145, 145), 10), ((255, 250), 15), ((200, 120), 6)]:
        cv2.circle(imagem, centro, raio, 25, -1)
    for centro, raio in [((48, 52), 6), ((350, 80), 8), ((70, 350), 5)]:
        cv2.circle(imagem, centro, raio, 255, -1)

    valor_otsu, binaria = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f"Limiar calculado por Otsu: {valor_otsu:.1f}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    erodida = cv2.erode(binaria, kernel, iterations=1)
    dilatada = cv2.dilate(binaria, kernel, iterations=1)
    abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    fechamento = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    limpa = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)

    resultados = {
        "01_binaria.png": binaria,
        "02_erodida.png": erodida,
        "03_dilatada.png": dilatada,
        "04_abertura.png": abertura,
        "05_fechamento.png": fechamento,
        "06_abertura_fechamento.png": limpa,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)
    painel = mosaico(
        [
            rotular(binaria, "Binaria com defeitos"),
            rotular(erodida, "Erosao"),
            rotular(dilatada, "Dilatacao"),
            rotular(abertura, "Abertura"),
            rotular(fechamento, "Fechamento"),
            rotular(limpa, "Abertura + fechamento"),
        ]
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap04", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
