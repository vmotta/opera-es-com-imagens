"""Capítulo 12: heatmaps, confiança, keypoints ausentes, ângulos e suavização temporal."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar

PONTOS = {
    "cabeca": (300, 85), "pescoco": (300, 145),
    "ombro_d": (235, 155), "cotovelo_d": (195, 235), "pulso_d": (170, 320),
    "ombro_e": (365, 155), "cotovelo_e": (405, 235), "pulso_e": (430, 320),
    "quadril_d": (265, 315), "joelho_d": (255, 440), "tornozelo_d": (245, 565),
    "quadril_e": (335, 315), "joelho_e": (345, 440), "tornozelo_e": (355, 565),
    "peito": (300, 235),
}

PARES = [
    ("cabeca", "pescoco"), ("pescoco", "ombro_d"), ("ombro_d", "cotovelo_d"),
    ("cotovelo_d", "pulso_d"), ("pescoco", "ombro_e"), ("ombro_e", "cotovelo_e"),
    ("cotovelo_e", "pulso_e"), ("pescoco", "peito"), ("peito", "quadril_d"),
    ("quadril_d", "joelho_d"), ("joelho_d", "tornozelo_d"), ("peito", "quadril_e"),
    ("quadril_e", "joelho_e"), ("joelho_e", "tornozelo_e"),
]


def heatmap_gaussiano(tamanho, centro, sigma=28, pico=1.0):
    altura, largura = tamanho
    y, x = np.mgrid[0:altura, 0:largura]
    cx, cy = centro
    mapa = pico * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return mapa.astype(np.float32)


def angulo(a, b, c):
    """Ângulo ABC em graus; devolve None em caso degenerado."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def desenhar_esqueleto(quadro, pontos, titulo):
    saida = quadro.copy()
    for a, b in PARES:
        pa, pb = pontos.get(a), pontos.get(b)
        if pa is not None and pb is not None:
            cv2.line(saida, pa, pb, (80, 230, 80), 5, cv2.LINE_AA)
    for nome, ponto in pontos.items():
        if ponto is not None:
            cv2.circle(saida, ponto, 8, (0, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(saida, nome, (ponto[0] + 7, ponto[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
    cv2.putText(saida, titulo, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
    return saida


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    quadro = np.full((640, 600, 3), 32, dtype=np.uint8)

    # --------------------------------------------------------------------------
    # 1. HEATMAPS COM DIFERENTES CONFIANÇAS.
    # O pulso esquerdo é propositalmente fraco para demonstrar keypoint ausente.
    # --------------------------------------------------------------------------
    picos = {nome: 1.0 for nome in PONTOS}
    picos["pulso_e"] = 0.12
    limiar = 0.20
    pontos_detectados = {}
    confiancas = {}
    heatmaps_vis = []

    for nome, centro_real in PONTOS.items():
        mapa = heatmap_gaussiano(quadro.shape[:2], centro_real, pico=picos[nome])
        _, confianca, _, ponto = cv2.minMaxLoc(mapa)
        confiancas[nome] = float(confianca)
        pontos_detectados[nome] = ponto if confianca >= limiar else None
        if nome in ("cabeca", "cotovelo_e", "pulso_e"):
            vis = cv2.normalize(mapa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.putText(vis, f"{nome}: max={confianca:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)
            heatmaps_vis.append(vis)

    for nome in ("cabeca", "cotovelo_e", "pulso_e"):
        print(f"{nome}: confiança={confiancas[nome]:.3f}; aceito={pontos_detectados[nome] is not None}")

    esqueleto = desenhar_esqueleto(quadro, pontos_detectados, f"Limiar de confiança = {limiar:.2f}")

    # --------------------------------------------------------------------------
    # 2. ÂNGULOS: o direito é válido; o esquerdo é indisponível pelo pulso ausente.
    # --------------------------------------------------------------------------
    ang_dir = angulo(
        pontos_detectados["ombro_d"],
        pontos_detectados["cotovelo_d"],
        pontos_detectados["pulso_d"],
    )
    print(f"Ângulo do cotovelo direito: {ang_dir:.2f}°" if ang_dir is not None else "Ângulo direito indisponível")
    if pontos_detectados["pulso_e"] is None:
        print("Ângulo do cotovelo esquerdo: indisponível porque o pulso ficou abaixo do limiar")

    if ang_dir is not None:
        p = pontos_detectados["cotovelo_d"]
        cv2.putText(esqueleto, f"{ang_dir:.1f} deg", (p[0] - 75, p[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 50), 2)

    # --------------------------------------------------------------------------
    # 3. CONVERSÃO DIDÁTICA DE COORDENADA DE HEATMAP PEQUENO PARA O FRAME.
    # --------------------------------------------------------------------------
    pequeno = heatmap_gaussiano((46, 46), (28, 18), sigma=3.0)
    _, _, _, ponto_pequeno = cv2.minMaxLoc(pequeno)
    x_img = round(ponto_pequeno[0] * quadro.shape[1] / 46)
    y_img = round(ponto_pequeno[1] * quadro.shape[0] / 46)
    print(f"Heatmap 46x46 ponto={ponto_pequeno} -> imagem ponto≈({x_img},{y_img})")

    # --------------------------------------------------------------------------
    # 4. SUAVIZAÇÃO TEMPORAL DE UM KEYPOINT COM RUÍDO.
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(123)
    verdadeiro = np.array(PONTOS["cotovelo_d"], dtype=np.float64)
    observacoes = verdadeiro + rng.normal(0, 7, size=(40, 2))
    alpha = 0.25
    exponencial = np.empty_like(observacoes)
    exponencial[0] = observacoes[0]
    for i in range(1, len(observacoes)):
        exponencial[i] = alpha * observacoes[i] + (1 - alpha) * exponencial[i - 1]

    erro_bruto = np.linalg.norm(observacoes - verdadeiro, axis=1)
    erro_suave = np.linalg.norm(exponencial - verdadeiro, axis=1)
    print(f"Erro médio bruto={erro_bruto.mean():.2f}px; suavizado={erro_suave.mean():.2f}px")

    grafico = np.full((300, 600, 3), 245, dtype=np.uint8)
    escala_x = 13
    for i in range(1, len(observacoes)):
        x0, x1 = 40 + (i - 1) * escala_x, 40 + i * escala_x
        yb0 = int(150 + (observacoes[i - 1, 0] - verdadeiro[0]) * 5)
        yb1 = int(150 + (observacoes[i, 0] - verdadeiro[0]) * 5)
        ys0 = int(150 + (exponencial[i - 1, 0] - verdadeiro[0]) * 5)
        ys1 = int(150 + (exponencial[i, 0] - verdadeiro[0]) * 5)
        cv2.line(grafico, (x0, yb0), (x1, yb1), (60, 60, 220), 2)
        cv2.line(grafico, (x0, ys0), (x1, ys1), (50, 170, 50), 2)
    cv2.putText(grafico, "vermelho: bruto | verde: filtro exponencial", (25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

    salvar(output_dir / "01_heatmap_cabeca.png", heatmaps_vis[0])
    salvar(output_dir / "02_esqueleto_simulado.png", esqueleto)
    salvar(output_dir / "03_heatmap_cotovelo.png", heatmaps_vis[1])
    salvar(output_dir / "04_heatmap_pulso_baixa_confianca.png", heatmaps_vis[2])
    salvar(output_dir / "05_suavizacao_temporal.png", grafico)

    painel = mosaico(
        [
            rotular(heatmaps_vis[0], "Heatmap cabeça"),
            rotular(heatmaps_vis[1], "Heatmap cotovelo"),
            rotular(heatmaps_vis[2], "Pulso: confiança baixa"),
            rotular(esqueleto, "Esqueleto com ausência"),
            rotular(grafico, "Ruído temporal e suavização"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap12", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
