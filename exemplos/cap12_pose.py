"""Capítulo 12: heatmaps simulados, confiança de keypoints e esqueleto."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar

PONTOS = {
    "cabeca": (300, 85),
    "pescoco": (300, 145),
    "ombro_d": (235, 155),
    "cotovelo_d": (195, 235),
    "pulso_d": (170, 320),
    "ombro_e": (365, 155),
    "cotovelo_e": (405, 235),
    "pulso_e": (430, 320),
    "quadril_d": (265, 315),
    "joelho_d": (255, 440),
    "tornozelo_d": (245, 565),
    "quadril_e": (335, 315),
    "joelho_e": (345, 440),
    "tornozelo_e": (355, 565),
    "peito": (300, 235),
}

PARES = [
    ("cabeca", "pescoco"),
    ("pescoco", "ombro_d"),
    ("ombro_d", "cotovelo_d"),
    ("cotovelo_d", "pulso_d"),
    ("pescoco", "ombro_e"),
    ("ombro_e", "cotovelo_e"),
    ("cotovelo_e", "pulso_e"),
    ("pescoco", "peito"),
    ("peito", "quadril_d"),
    ("quadril_d", "joelho_d"),
    ("joelho_d", "tornozelo_d"),
    ("peito", "quadril_e"),
    ("quadril_e", "joelho_e"),
    ("joelho_e", "tornozelo_e"),
]


def heatmap_gaussiano(tamanho, centro, sigma=28):
    altura, largura = tamanho
    y, x = np.mgrid[0:altura, 0:largura]
    cx, cy = centro
    mapa = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return np.uint8(mapa * 255)


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    quadro = np.full((640, 600, 3), 32, dtype=np.uint8)
    pontos_detectados: dict[str, tuple[int, int] | None] = {}

    # Em uma rede real, cada parte possui um heatmap. O máximo fornece a posição.
    heatmap_cabeca = heatmap_gaussiano(quadro.shape[:2], PONTOS["cabeca"])
    for nome, centro_real in PONTOS.items():
        mapa = heatmap_gaussiano(quadro.shape[:2], centro_real)
        _, confianca, _, ponto = cv2.minMaxLoc(mapa.astype(np.float32) / 255.0)
        pontos_detectados[nome] = ponto if confianca >= 0.10 else None

    for parte_a, parte_b in PARES:
        ponto_a, ponto_b = pontos_detectados[parte_a], pontos_detectados[parte_b]
        if ponto_a is not None and ponto_b is not None:
            cv2.line(quadro, ponto_a, ponto_b, (80, 230, 80), 5, cv2.LINE_AA)
    for nome, ponto in pontos_detectados.items():
        if ponto is not None:
            cv2.circle(quadro, ponto, 8, (0, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(quadro, nome, (ponto[0] + 8, ponto[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)

    salvar(output_dir / "01_heatmap_cabeca.png", heatmap_cabeca)
    salvar(output_dir / "02_esqueleto_simulado.png", quadro)


def main() -> None:
    parser = criar_parser("cap12", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
