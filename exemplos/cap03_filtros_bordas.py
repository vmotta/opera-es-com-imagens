"""Capítulo 3: ruído, kernels, filtros espaciais, Sobel, magnitude e Canny."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def adicionar_sal_pimenta(imagem: np.ndarray, proporcao: float = 0.05) -> np.ndarray:
    """Adiciona ruído impulsivo reprodutível para comparação entre filtros."""
    gerador = np.random.default_rng(42)
    mapa = gerador.random(imagem.shape)
    saida = imagem.copy()
    saida[mapa < proporcao / 2] = 0
    saida[mapa > 1 - proporcao / 2] = 255
    return saida


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # Cena em tons de cinza com regiões constantes e bordas bem definidas.
    limpa = np.full((360, 480), 128, dtype=np.uint8)
    cv2.rectangle(limpa, (50, 55), (250, 255), 45, -1)
    cv2.circle(limpa, (345, 215), 90, 210, -1)
    cv2.line(limpa, (30, 315), (450, 315), 240, 6)

    ruidosa = adicionar_sal_pimenta(limpa)

    # --------------------------------------------------------------------------
    # 1. KERNEL DE MÉDIA MANUAL E FUNÇÕES PRONTAS.
    # --------------------------------------------------------------------------
    kernel_media = np.ones((7, 7), dtype=np.float32) / 49.0
    media_manual = cv2.filter2D(ruidosa, -1, kernel_media)
    media = cv2.blur(ruidosa, (7, 7))
    print("Diferença máxima filter2D média vs cv2.blur:", int(cv2.absdiff(media_manual, media).max()))

    gaussiana = cv2.GaussianBlur(ruidosa, (7, 7), 1.4)
    mediana = cv2.medianBlur(ruidosa, 5)
    bilateral = cv2.bilateralFilter(ruidosa, d=9, sigmaColor=60, sigmaSpace=60)

    # --------------------------------------------------------------------------
    # 2. KERNEL DE REALCE.
    # --------------------------------------------------------------------------
    kernel_realce = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    realcada = cv2.filter2D(limpa, -1, kernel_realce)

    # --------------------------------------------------------------------------
    # 3. DERIVADAS SOBEL EM TIPO COM SINAL.
    # --------------------------------------------------------------------------
    sobel_x_64f = cv2.Sobel(limpa, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_64f = cv2.Sobel(limpa, cv2.CV_64F, 0, 1, ksize=3)
    print(
        "Sobel X intervalo antes da conversão:",
        float(sobel_x_64f.min()),
        "até",
        float(sobel_x_64f.max()),
    )

    sobel_x = cv2.convertScaleAbs(sobel_x_64f)
    sobel_y = cv2.convertScaleAbs(sobel_y_64f)

    # Magnitude geométrica do gradiente.
    magnitude = cv2.magnitude(
        sobel_x_64f.astype(np.float32),
        sobel_y_64f.astype(np.float32),
    )
    magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --------------------------------------------------------------------------
    # 4. CANNY: sem e com suavização para observar sensibilidade ao ruído.
    # --------------------------------------------------------------------------
    canny_sem_suavizar = cv2.Canny(ruidosa, 60, 150)
    canny = cv2.Canny(cv2.GaussianBlur(ruidosa, (5, 5), 1.0), 60, 150)
    print("Pixels de borda sem suavização:", int(np.count_nonzero(canny_sem_suavizar)))
    print("Pixels de borda com suavização:", int(np.count_nonzero(canny)))

    resultados = {
        "01_limpa.png": limpa,
        "02_ruidosa.png": ruidosa,
        "03_media.png": media,
        "04_gaussiana.png": gaussiana,
        "05_mediana.png": mediana,
        "06_bilateral.png": bilateral,
        "07_realce.png": realcada,
        "08_sobel_x.png": sobel_x,
        "09_sobel_y.png": sobel_y,
        "10_magnitude.png": magnitude_vis,
        "11_canny_sem_suavizacao.png": canny_sem_suavizar,
        "12_canny.png": canny,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)

    painel = mosaico(
        [
            rotular(ruidosa, "Sal e pimenta"),
            rotular(media, "Media 7x7"),
            rotular(gaussiana, "Gaussiano 7x7"),
            rotular(mediana, "Mediana 5"),
            rotular(bilateral, "Bilateral"),
            rotular(realcada, "Kernel de realce"),
            rotular(sobel_x, "Sobel X"),
            rotular(sobel_y, "Sobel Y"),
            rotular(magnitude_vis, "Magnitude do gradiente"),
            rotular(canny_sem_suavizar, "Canny no ruido"),
            rotular(canny, "Canny apos Gaussiano"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap03", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
