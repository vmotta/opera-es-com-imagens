"""Capítulo 3: ruído, filtros espaciais, Sobel e Canny."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    limpa = np.full((360, 480), 128, dtype=np.uint8)
    cv2.rectangle(limpa, (50, 55), (250, 255), 45, -1)
    cv2.circle(limpa, (345, 215), 90, 210, -1)

    gerador = np.random.default_rng(42)
    mapa = gerador.random(limpa.shape)
    ruidosa = limpa.copy()
    ruidosa[mapa < 0.025] = 0
    ruidosa[mapa > 0.975] = 255

    media = cv2.blur(ruidosa, (7, 7))
    gaussiana = cv2.GaussianBlur(ruidosa, (7, 7), 0)
    mediana = cv2.medianBlur(ruidosa, 5)

    sobel_x_64f = cv2.Sobel(limpa, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_64f = cv2.Sobel(limpa, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x_64f)
    sobel_y = cv2.convertScaleAbs(sobel_y_64f)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    canny = cv2.Canny(cv2.GaussianBlur(limpa, (5, 5), 0), 60, 150)

    resultados = {
        "01_ruidosa.png": ruidosa,
        "02_media.png": media,
        "03_gaussiana.png": gaussiana,
        "04_mediana.png": mediana,
        "05_sobel.png": sobel,
        "06_canny.png": canny,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)

    painel = mosaico(
        [
            rotular(ruidosa, "Sal e pimenta"),
            rotular(media, "Media 7x7"),
            rotular(gaussiana, "Gaussiano 7x7"),
            rotular(mediana, "Mediana 5"),
            rotular(sobel, "Sobel X + Y"),
            rotular(canny, "Canny 60/150"),
        ]
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap03", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
