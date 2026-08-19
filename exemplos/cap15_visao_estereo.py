"""Capítulo 15: estéreo sintético, SGBM, disparidade métrica e sensibilidade da profundidade."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def profundidade_por_disparidade(disparidade, focal_px, baseline_m):
    disparidade = np.asarray(disparidade, dtype=np.float64)
    resultado = np.full(disparidade.shape, np.nan, dtype=np.float64)
    validos = disparidade > 0
    resultado[validos] = focal_px * baseline_m / disparidade[validos]
    return resultado


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    rng = np.random.default_rng(15)

    # Textura compartilhada: fornece padrões locais para correspondência.
    textura = rng.integers(25, 85, (320, 640), dtype=np.uint8)
    esquerda = textura.copy()
    direita = textura.copy()

    # Três regiões com disparidades conhecidas.
    objetos = [
        ("circulo_proximo", 56),
        ("retangulo_medio", 32),
        ("retangulo_distante", 16),
    ]
    cv2.circle(esquerda, (190, 175), 50, 225, -1)
    cv2.circle(direita, (190 - 56, 175), 50, 225, -1)
    cv2.rectangle(esquerda, (315, 85), (390, 250), 185, -1)
    cv2.rectangle(direita, (315 - 32, 85), (390 - 32, 250), 185, -1)
    cv2.rectangle(esquerda, (500, 65), (565, 270), 155, -1)
    cv2.rectangle(direita, (500 - 16, 65), (565 - 16, 270), 155, -1)

    # Região uniforme para mostrar um caso difícil de correspondência.
    cv2.rectangle(esquerda, (10, 260), (150, 315), 110, -1)
    cv2.rectangle(direita, (10, 260), (150, 315), 110, -1)

    numero_disparidades = 16 * 8
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

    # --------------------------------------------------------------------------
    # 1. IMPORTANTE: saída SGBM em ponto fixo; /16 recupera pixels.
    # --------------------------------------------------------------------------
    disparidade_fixa = stereo.compute(esquerda, direita)
    disparidade = disparidade_fixa.astype(np.float32) / 16.0
    validos = disparidade > 0
    print(
        f"Disparidade fixa dtype={disparidade_fixa.dtype}; após /16 dtype={disparidade.dtype}; "
        f"pixels válidos={int(validos.sum())}/{validos.size}"
    )

    # Visualização é uma cópia normalizada; NÃO será usada no cálculo métrico.
    visual = np.zeros_like(esquerda)
    if np.any(validos):
        min_d = float(disparidade[validos].min())
        max_d = float(disparidade[validos].max())
        normalizado = (disparidade[validos] - min_d) / max(1e-6, max_d - min_d)
        visual[validos] = np.clip(normalizado * 255, 0, 255).astype(np.uint8)
        print(f"Faixa de disparidade válida: {min_d:.2f}..{max_d:.2f} pixels")
    colorido = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)

    # --------------------------------------------------------------------------
    # 2. PROFUNDIDADE MÉTRICA USANDO DISPARIDADE ORIGINAL.
    # --------------------------------------------------------------------------
    baseline_m = 0.12
    focal_px = 700.0
    profundidade = profundidade_por_disparidade(disparidade, focal_px, baseline_m).astype(np.float32)

    for nome, d in objetos:
        z = focal_px * baseline_m / d
        z_menos = focal_px * baseline_m / max(1, d - 1)
        z_mais = focal_px * baseline_m / (d + 1)
        print(
            f"{nome}: d={d}px -> Z={z:.3f}m; "
            f"se d-1 -> {z_menos:.3f}m; se d+1 -> {z_mais:.3f}m"
        )

    # Visualização de profundidade apenas para inspeção.
    profundidade_vis = np.zeros_like(esquerda)
    finitos = np.isfinite(profundidade) & (profundidade > 0) & (profundidade < 15)
    if np.any(finitos):
        valores = profundidade[finitos]
        p5, p95 = np.percentile(valores, [5, 95])
        norm = 1 - np.clip((valores - p5) / max(1e-6, p95 - p5), 0, 1)
        profundidade_vis[finitos] = (norm * 255).astype(np.uint8)
    profundidade_colorida = cv2.applyColorMap(profundidade_vis, cv2.COLORMAP_TURBO)

    # --------------------------------------------------------------------------
    # 3. CURVA Z=fB/d: evidencia a relação inversa e sensibilidade em d pequeno.
    # --------------------------------------------------------------------------
    ds = np.arange(4, 97, dtype=np.float64)
    zs = focal_px * baseline_m / ds
    grafico = np.full((360, 720, 3), 248, dtype=np.uint8)
    x0, y0 = 55, 310
    largura_plot, altura_plot = 620, 250
    z_max = float(zs.max())
    pontos = []
    for d, z in zip(ds, zs, strict=True):
        x = int(x0 + (d - ds.min()) / (ds.max() - ds.min()) * largura_plot)
        y = int(y0 - z / z_max * altura_plot)
        pontos.append((x, y))
    cv2.polylines(grafico, [np.array(pontos, dtype=np.int32)], False, (40, 80, 210), 3)
    cv2.putText(grafico, "Z = fB/d: profundidade cai quando disparidade cresce", (35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2)
    cv2.putText(grafico, "d (pixels) ->", (560, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

    salvar(output_dir / "01_esquerda.png", esquerda)
    salvar(output_dir / "02_direita.png", direita)
    salvar(output_dir / "03_mapa_disparidade.png", colorido)
    salvar(output_dir / "04_mapa_profundidade_visual.png", profundidade_colorida)
    salvar(output_dir / "05_curva_disparidade_profundidade.png", grafico)

    painel = mosaico(
        [
            rotular(esquerda, "Camera esquerda"),
            rotular(direita, "Camera direita"),
            rotular(colorido, "Disparidade visualizada"),
            rotular(profundidade_colorida, "Profundidade visual"),
            rotular(grafico, "Z=fB/d"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap15", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
