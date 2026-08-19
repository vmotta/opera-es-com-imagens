"""Capítulo 9: pontos Shi-Tomasi e fluxo óptico esparso Lucas-Kanade."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    frame_t = np.full((330, 480, 3), 45, dtype=np.uint8)
    frame_t1 = np.full_like(frame_t, 45)

    cv2.rectangle(frame_t, (90, 170), (175, 245), (0, 230, 0), -1)
    cv2.line(frame_t, (98, 180), (165, 235), (0, 80, 0), 5)
    cv2.circle(frame_t, (135, 205), 12, (255, 255, 255), -1)

    dx, dy = 68, -42
    cv2.rectangle(frame_t1, (90 + dx, 170 + dy), (175 + dx, 245 + dy), (0, 230, 0), -1)
    cv2.line(frame_t1, (98 + dx, 180 + dy), (165 + dx, 235 + dy), (0, 80, 0), 5)
    cv2.circle(frame_t1, (135 + dx, 205 + dy), 12, (255, 255, 255), -1)

    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    gray_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
    pontos_t = cv2.goodFeaturesToTrack(gray_t, maxCorners=30, qualityLevel=0.15, minDistance=7)
    if pontos_t is None:
        raise RuntimeError("Nenhuma quina adequada foi encontrada para rastreamento.")

    parametros_lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    }
    pontos_t1, status, erros = cv2.calcOpticalFlowPyrLK(
        gray_t,
        gray_t1,
        pontos_t,
        None,
        **parametros_lk,
    )
    if pontos_t1 is None or status is None:
        raise RuntimeError("Lucas-Kanade não conseguiu estimar o deslocamento.")

    validos = status.ravel() == 1
    novos = pontos_t1[validos].reshape(-1, 2)
    antigos = pontos_t[validos].reshape(-1, 2)
    erros_validos = erros[validos].reshape(-1) if erros is not None else np.zeros(len(novos))
    saida = frame_t1.copy()
    deslocamentos: list[np.ndarray] = []
    for novo, antigo, erro in zip(novos, antigos, erros_validos, strict=True):
        if erro > 20:
            continue
        novo_i = tuple(np.round(novo).astype(int))
        antigo_i = tuple(np.round(antigo).astype(int))
        cv2.arrowedLine(saida, antigo_i, novo_i, (0, 255, 255), 2, tipLength=0.22)
        cv2.circle(saida, novo_i, 5, (0, 0, 255), -1)
        deslocamentos.append(novo - antigo)

    if deslocamentos:
        mediana = np.median(np.vstack(deslocamentos), axis=0)
        print(f"Deslocamento mediano estimado: dx={mediana[0]:.1f}, dy={mediana[1]:.1f}")
        print(f"Deslocamento real: dx={dx}, dy={dy}")

    salvar(output_dir / "01_frame_t.png", frame_t)
    salvar(output_dir / "02_frame_t1.png", frame_t1)
    salvar(output_dir / "03_vetores_lucas_kanade.png", saida)


def main() -> None:
    parser = criar_parser("cap09", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
