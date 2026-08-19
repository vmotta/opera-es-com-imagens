"""Capítulo 6: conversão HSV e segmentação cromática robusta."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.full((360, 640, 3), 235, dtype=np.uint8)
    cv2.circle(imagem, (130, 180), 85, (255, 30, 30), -1)  # azul em BGR
    cv2.circle(imagem, (320, 180), 85, (30, 220, 30), -1)
    cv2.circle(imagem, (510, 180), 85, (20, 20, 235), -1)

    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    limite_inferior = np.array([100, 80, 50], dtype=np.uint8)
    limite_superior = np.array([140, 255, 255], dtype=np.uint8)
    mascara_azul = cv2.inRange(hsv, limite_inferior, limite_superior)

    # A faixa vermelha cruza o início/fim do círculo de matiz do OpenCV.
    vermelho_baixo = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
    vermelho_alto = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([179, 255, 255]))
    mascara_vermelha = cv2.bitwise_or(vermelho_baixo, vermelho_alto)

    resultado_azul = cv2.bitwise_and(imagem, imagem, mask=mascara_azul)
    resultado_vermelho = cv2.bitwise_and(imagem, imagem, mask=mascara_vermelha)

    salvar(output_dir / "01_original_bgr.png", imagem)
    salvar(output_dir / "02_mascara_azul.png", mascara_azul)
    salvar(output_dir / "03_resultado_azul.png", resultado_azul)
    salvar(output_dir / "04_mascara_vermelha_dupla.png", mascara_vermelha)
    salvar(output_dir / "05_resultado_vermelho.png", resultado_vermelho)
    painel = mosaico(
        [
            rotular(imagem, "Cena BGR"),
            rotular(mascara_azul, "Mascara H=100..140"),
            rotular(resultado_azul, "Azul isolado"),
            rotular(mascara_vermelha, "Vermelho: 2 faixas"),
            rotular(resultado_vermelho, "Vermelho isolado"),
        ]
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap06", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
