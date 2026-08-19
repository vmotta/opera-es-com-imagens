"""Capítulo 4: limiar fixo, Otsu, adaptativo e operações morfológicas."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # Fundo com gradiente para mostrar a limitação de um único limiar global.
    gradiente = np.linspace(25, 105, 480, dtype=np.uint8)
    imagem = np.tile(gradiente, (400, 1))
    cv2.rectangle(imagem, (105, 90), (365, 310), 205, -1)

    # Buracos escuros dentro do objeto.
    for centro, raio in [((155, 145), 10), ((285, 250), 15), ((220, 125), 6)]:
        cv2.circle(imagem, centro, raio, 30, -1)

    # Pontos claros no fundo.
    for centro, raio in [((48, 52), 6), ((430, 80), 8), ((70, 350), 5), ((410, 335), 7)]:
        cv2.circle(imagem, centro, raio, 255, -1)

    # --------------------------------------------------------------------------
    # 1. COMPARAÇÃO ENTRE LIMIAR FIXO, OTSU E ADAPTATIVO.
    # --------------------------------------------------------------------------
    _, limiar_fixo = cv2.threshold(imagem, 140, 255, cv2.THRESH_BINARY)

    suave = cv2.GaussianBlur(imagem, (5, 5), 0)
    valor_otsu, binaria_otsu = cv2.threshold(
        suave,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    print(f"Limiar calculado por Otsu: {valor_otsu:.1f}")

    adaptativa = cv2.adaptiveThreshold(
        imagem,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    # Para as operações seguintes, usamos Otsu como máscara base.
    binaria = binaria_otsu

    # --------------------------------------------------------------------------
    # 2. ELEMENTOS ESTRUTURANTES.
    # --------------------------------------------------------------------------
    kernel_elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_cruz = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    print("Kernel elíptico:\n", kernel_elipse)
    print("Kernel em cruz:\n", kernel_cruz)

    # --------------------------------------------------------------------------
    # 3. EROSÃO, DILATAÇÃO, ABERTURA, FECHAMENTO E GRADIENTE.
    # --------------------------------------------------------------------------
    erodida = cv2.erode(binaria, kernel_elipse, iterations=1)
    dilatada = cv2.dilate(binaria, kernel_elipse, iterations=1)
    abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_elipse)
    fechamento = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_elipse)
    gradiente_morfologico = cv2.morphologyEx(binaria, cv2.MORPH_GRADIENT, kernel_elipse)
    limpa = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel_elipse)

    # --------------------------------------------------------------------------
    # 4. TOP-HAT E BLACK-HAT EM ESCALA DE CINZA.
    # --------------------------------------------------------------------------
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    top_hat = cv2.morphologyEx(imagem, cv2.MORPH_TOPHAT, kernel_grande)
    black_hat = cv2.morphologyEx(imagem, cv2.MORPH_BLACKHAT, kernel_grande)

    resultados = {
        "01_imagem_entrada.png": imagem,
        "02_limiar_fixo.png": limiar_fixo,
        "03_otsu.png": binaria_otsu,
        "04_adaptativo.png": adaptativa,
        "05_erodida.png": erodida,
        "06_dilatada.png": dilatada,
        "07_abertura.png": abertura,
        "08_fechamento.png": fechamento,
        "09_gradiente_morfologico.png": gradiente_morfologico,
        "10_abertura_fechamento.png": limpa,
        "11_top_hat.png": top_hat,
        "12_black_hat.png": black_hat,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)

    painel = mosaico(
        [
            rotular(imagem, "Entrada com gradiente"),
            rotular(limiar_fixo, "Limiar fixo 140"),
            rotular(binaria_otsu, f"Otsu T={valor_otsu:.0f}"),
            rotular(adaptativa, "Limiar adaptativo"),
            rotular(erodida, "Erosao"),
            rotular(dilatada, "Dilatacao"),
            rotular(abertura, "Abertura"),
            rotular(fechamento, "Fechamento"),
            rotular(gradiente_morfologico, "Gradiente morfologico"),
            rotular(limpa, "Abertura + fechamento"),
            rotular(top_hat, "Top-hat"),
            rotular(black_hat, "Black-hat"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap04", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
