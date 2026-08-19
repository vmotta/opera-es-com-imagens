"""Capítulo 7: Template Matching, métricas, busca multiescala e Haar Cascade."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def normalizar_mapa(mapa: np.ndarray, inverter: bool = False) -> np.ndarray:
    """Converte mapa de similaridade/erro para uint8 apenas para visualização."""
    vis = cv2.normalize(mapa, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return 255 - vis if inverter else vis


def melhor_match_multiescala(
    cena: np.ndarray,
    template: np.ndarray,
    escalas: tuple[float, ...],
) -> tuple[float, tuple[int, int], tuple[int, int], float]:
    """Busca o template em várias escalas e devolve o melhor resultado CCOEFF_NORMED."""
    melhor_score = -1.0
    melhor_local = (0, 0)
    melhor_tamanho = template.shape[1], template.shape[0]
    melhor_escala = 1.0

    for escala in escalas:
        largura = max(1, round(template.shape[1] * escala))
        altura = max(1, round(template.shape[0] * escala))
        if largura > cena.shape[1] or altura > cena.shape[0]:
            continue

        red = cv2.resize(template, (largura, altura), interpolation=cv2.INTER_LINEAR)
        mapa = cv2.matchTemplate(cena, red, cv2.TM_CCOEFF_NORMED)
        _, score, _, local = cv2.minMaxLoc(mapa)
        print(f"escala={escala:.2f} score={score:.4f} local={local}")

        if score > melhor_score:
            melhor_score = float(score)
            melhor_local = local
            melhor_tamanho = (largura, altura)
            melhor_escala = escala

    return melhor_score, melhor_local, melhor_tamanho, melhor_escala


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # --------------------------------------------------------------------------
    # 1. CENA E TEMPLATE.
    # --------------------------------------------------------------------------
    cena = np.full((380, 560, 3), 155, dtype=np.uint8)
    cv2.circle(cena, (65, 65), 22, (0, 0, 255), -1)
    cv2.rectangle(cena, (110, 260), (155, 305), (255, 0, 0), -1)
    cv2.circle(cena, (385, 190), 44, (0, 255, 255), -1)
    cv2.line(cena, (315, 55), (480, 80), (20, 20, 20), 3)

    template = np.full((72, 72, 3), 155, dtype=np.uint8)
    cv2.circle(template, (36, 36), 34, (0, 255, 255), -1)

    # --------------------------------------------------------------------------
    # 2. DUAS MÉTRICAS: CCOEFF (máximo) E SQDIFF (mínimo).
    # --------------------------------------------------------------------------
    mapa_ccoeff = cv2.matchTemplate(cena, template, cv2.TM_CCOEFF_NORMED)
    _, max_c, _, max_loc_c = cv2.minMaxLoc(mapa_ccoeff)

    mapa_sqdiff = cv2.matchTemplate(cena, template, cv2.TM_SQDIFF_NORMED)
    min_s, _, min_loc_s, _ = cv2.minMaxLoc(mapa_sqdiff)

    print(f"TM_CCOEFF_NORMED: melhor=max={max_c:.4f} em {max_loc_c}")
    print(f"TM_SQDIFF_NORMED: melhor=min={min_s:.4f} em {min_loc_s}")

    mapa_ccoeff_vis = normalizar_mapa(mapa_ccoeff)
    mapa_sqdiff_vis = normalizar_mapa(mapa_sqdiff, inverter=True)

    # --------------------------------------------------------------------------
    # 3. BUSCA MULTIESCALA.
    # --------------------------------------------------------------------------
    escalas = (0.75, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40)
    score, local, tamanho, escala = melhor_match_multiescala(cena, template, escalas)
    marcado = cena.copy()
    x, y = local
    w, h = tamanho
    if score >= 0.60:
        cv2.rectangle(marcado, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            marcado,
            f"esc={escala:.2f} score={score:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
        )
    print(f"Melhor multiescala: escala={escala:.2f}, score={score:.4f}")

    # --------------------------------------------------------------------------
    # 4. HAAR CASCADE: CARREGAMENTO E VARIAÇÃO DE minNeighbors.
    # --------------------------------------------------------------------------
    rosto_didatico = np.full((420, 420, 3), 245, dtype=np.uint8)
    cv2.ellipse(rosto_didatico, (210, 215), (120, 155), 0, 0, 360, (150, 190, 225), -1)
    cv2.circle(rosto_didatico, (165, 185), 14, (30, 30, 30), -1)
    cv2.circle(rosto_didatico, (255, 185), 14, (30, 30, 30), -1)
    cv2.ellipse(rosto_didatico, (210, 265), (50, 24), 0, 0, 180, (30, 30, 30), 5)

    caminho_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascata = cv2.CascadeClassifier(caminho_xml)
    if cascata.empty():
        raise RuntimeError(f"Classificador Haar não carregado: {caminho_xml}")

    cinza = cv2.cvtColor(rosto_didatico, cv2.COLOR_BGR2GRAY)
    resultados_haar: list[np.ndarray] = []
    for min_neighbors in (2, 5, 8):
        visual = rosto_didatico.copy()
        rostos = cascata.detectMultiScale(
            cinza,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=(40, 40),
        )
        for rx, ry, rw, rh in rostos:
            cv2.rectangle(visual, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
        cv2.putText(
            visual,
            f"minNeighbors={min_neighbors}: {len(rostos)}",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 180),
            2,
        )
        print(f"Haar minNeighbors={min_neighbors}: {len(rostos)} detecções")
        resultados_haar.append(visual)

    salvar(output_dir / "01_template.png", template)
    salvar(output_dir / "02_mapa_ccoeff.png", mapa_ccoeff_vis)
    salvar(output_dir / "03_mapa_sqdiff_invertido.png", mapa_sqdiff_vis)
    salvar(output_dir / "04_template_multiescala.png", marcado)
    for i, visual in enumerate(resultados_haar, start=1):
        salvar(output_dir / f"0{4+i}_haar_parametro_{i}.png", visual)

    salvar(output_dir / "02_template_encontrado.png", marcado)
    salvar(output_dir / "03_haar_experimento.png", resultados_haar[1])

    painel = mosaico(
        [
            rotular(cena, "Cena"),
            rotular(template, "Template"),
            rotular(mapa_ccoeff_vis, "CCOEFF: max e melhor"),
            rotular(mapa_sqdiff_vis, "SQDIFF invertido"),
            rotular(marcado, "Busca multiescala"),
            rotular(resultados_haar[0], "Haar neighbors=2"),
            rotular(resultados_haar[1], "Haar neighbors=5"),
            rotular(resultados_haar[2], "Haar neighbors=8"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap07", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
