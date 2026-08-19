"""Capítulo 15: par estéreo sintético, disparidade SGBM e profundidade."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    gerador = np.random.default_rng(15)
    textura = gerador.integers(25, 85, (320, 640), dtype=np.uint8)
    esquerda = textura.copy()
    direita = textura.copy()

    # O mesmo objeto aparece mais à esquerda na câmera direita. Quanto maior o
    # deslocamento, maior a disparidade e menor a profundidade estimada.
    cv2.circle(esquerda, (220, 175), 55, 220, -1)
    cv2.circle(direita, (164, 175), 55, 220, -1)  # disparidade 56 px
    cv2.rectangle(esquerda, (455, 70), (535, 270), 165, -1)
    cv2.rectangle(direita, (439, 70), (519, 270), 165, -1)  # disparidade 16 px

    numero_disparidades = 16 * 6
    bloco = 7
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numero_disparidades,
        blockSize=bloco,
        P1=8 * bloco**2,
        P2=32 * bloco**2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparidade = stereo.compute(esquerda, direita).astype(np.float32) / 16.0
    validos = disparidade > 0
    visual = np.zeros_like(esquerda)
    if np.any(validos):
        visual[validos] = cv2.normalize(
            disparidade[validos],
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).reshape(-1).astype(np.uint8)
    colorido = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)

    baseline_m = 0.12
    focal_px = 700.0
    for nome, d in [("círculo próximo", 56), ("retângulo distante", 16)]:
        profundidade = focal_px * baseline_m / d
        print(f"{nome}: Z = fB/d = {focal_px:.0f}*{baseline_m:.2f}/{d} = {profundidade:.2f} m")

    salvar(output_dir / "01_esquerda.png", esquerda)
    salvar(output_dir / "02_direita.png", direita)
    salvar(output_dir / "03_mapa_disparidade.png", colorido)
    painel = mosaico(
        [
            rotular(esquerda, "Camera esquerda"),
            rotular(direita, "Camera direita"),
            rotular(colorido, "Disparidade (claro = maior)"),
        ]
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap15", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
