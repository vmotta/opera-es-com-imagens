"""Capítulo 9: Shi-Tomasi, Lucas-Kanade, ida-volta e velocidade em vídeo."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def criar_frame(tamanho: tuple[int, int], deslocamento=(0, 0), ocluir=False) -> np.ndarray:
    altura, largura = tamanho
    dx, dy = deslocamento
    frame = np.full((altura, largura, 3), 45, dtype=np.uint8)
    cv2.rectangle(frame, (90 + dx, 170 + dy), (175 + dx, 245 + dy), (0, 230, 0), -1)
    cv2.line(frame, (98 + dx, 180 + dy), (165 + dx, 235 + dy), (0, 80, 0), 5)
    cv2.circle(frame, (135 + dx, 205 + dy), 12, (255, 255, 255), -1)
    cv2.rectangle(frame, (110 + dx, 190 + dy), (125 + dx, 205 + dy), (255, 0, 0), -1)
    if ocluir:
        cv2.rectangle(frame, (145 + dx, 160 + dy), (205 + dx, 220 + dy), (45, 45, 45), -1)
    return frame


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    altura, largura = 330, 480
    dx_real, dy_real = 68, -42
    fps = 30.0

    frame_t = criar_frame((altura, largura))
    frame_t1 = criar_frame((altura, largura), (dx_real, dy_real))
    frame_ocluido = criar_frame((altura, largura), (dx_real, dy_real), ocluir=True)

    gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    gray_t1 = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------------------------
    # 1. PONTOS SHI-TOMASI.
    # --------------------------------------------------------------------------
    pontos_t = cv2.goodFeaturesToTrack(
        gray_t,
        maxCorners=50,
        qualityLevel=0.05,
        minDistance=7,
        blockSize=7,
    )
    if pontos_t is None:
        raise RuntimeError("Nenhuma quina adequada foi encontrada para rastreamento.")

    pontos_vis = frame_t.copy()
    for ponto in pontos_t.reshape(-1, 2):
        p = tuple(np.round(ponto).astype(int))
        cv2.circle(pontos_vis, p, 5, (0, 0, 255), -1)
    print(f"Shi-Tomasi encontrou {len(pontos_t)} pontos")

    parametros_lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }

    # --------------------------------------------------------------------------
    # 2. FLUXO t -> t+1.
    # --------------------------------------------------------------------------
    pontos_t1, status_ida, erros_ida = cv2.calcOpticalFlowPyrLK(
        gray_t, gray_t1, pontos_t, None, **parametros_lk
    )
    if pontos_t1 is None or status_ida is None:
        raise RuntimeError("Lucas-Kanade não conseguiu estimar o deslocamento.")

    # --------------------------------------------------------------------------
    # 3. VERIFICAÇÃO IDA-VOLTA t -> t+1 -> t.
    # --------------------------------------------------------------------------
    pontos_retorno, status_volta, _ = cv2.calcOpticalFlowPyrLK(
        gray_t1, gray_t, pontos_t1, None, **parametros_lk
    )
    if pontos_retorno is None or status_volta is None:
        raise RuntimeError("Fluxo de retorno não pôde ser calculado.")

    erro_fb = np.linalg.norm(
        pontos_retorno.reshape(-1, 2) - pontos_t.reshape(-1, 2),
        axis=1,
    )
    erro_lk = erros_ida.reshape(-1) if erros_ida is not None else np.zeros(len(pontos_t))

    validos = (
        (status_ida.reshape(-1) == 1)
        & (status_volta.reshape(-1) == 1)
        & (erro_lk < 20)
        & (erro_fb < 1.5)
    )
    print(f"Pontos aprovados após ida-volta: {int(validos.sum())}/{len(validos)}")
    print(
        "Erro ida-volta: mediana=",
        f"{np.median(erro_fb):.3f}",
        "máximo=",
        f"{np.max(erro_fb):.3f}",
    )

    novos = pontos_t1.reshape(-1, 2)[validos]
    antigos = pontos_t.reshape(-1, 2)[validos]
    saida = frame_t1.copy()
    deslocamentos = novos - antigos

    for novo, antigo in zip(novos, antigos, strict=True):
        novo_i = tuple(np.round(novo).astype(int))
        antigo_i = tuple(np.round(antigo).astype(int))
        cv2.arrowedLine(saida, antigo_i, novo_i, (0, 255, 255), 2, tipLength=0.22)
        cv2.circle(saida, novo_i, 5, (0, 0, 255), -1)

    if len(deslocamentos):
        media = np.mean(deslocamentos, axis=0)
        mediana = np.median(deslocamentos, axis=0)
        velocidade = mediana * fps
        print(f"Deslocamento médio: dx={media[0]:.2f}, dy={media[1]:.2f} px/frame")
        print(f"Deslocamento mediano: dx={mediana[0]:.2f}, dy={mediana[1]:.2f} px/frame")
        print(f"Deslocamento real: dx={dx_real}, dy={dy_real} px/frame")
        print(f"Velocidade mediana: vx={velocidade[0]:.1f}, vy={velocidade[1]:.1f} px/s a {fps:.0f} FPS")

    # --------------------------------------------------------------------------
    # 4. EXPERIMENTO COM OCLUSÃO: os mesmos pontos enfrentam área escondida.
    # --------------------------------------------------------------------------
    gray_ocluido = cv2.cvtColor(frame_ocluido, cv2.COLOR_BGR2GRAY)
    pontos_ocluido, status_occ, erros_occ = cv2.calcOpticalFlowPyrLK(
        gray_t, gray_ocluido, pontos_t, None, **parametros_lk
    )
    visual_oclusao = frame_ocluido.copy()
    if pontos_ocluido is not None and status_occ is not None:
        for i, (origem, destino) in enumerate(zip(pontos_t.reshape(-1, 2), pontos_ocluido.reshape(-1, 2), strict=True)):
            ok = bool(status_occ.reshape(-1)[i])
            erro = float(erros_occ.reshape(-1)[i]) if erros_occ is not None else 0.0
            cor = (0, 255, 0) if ok and erro < 20 else (0, 0, 255)
            cv2.line(
                visual_oclusao,
                tuple(np.round(origem).astype(int)),
                tuple(np.round(destino).astype(int)),
                cor,
                1,
            )

    salvar(output_dir / "01_frame_t.png", frame_t)
    salvar(output_dir / "02_pontos_shi_tomasi.png", pontos_vis)
    salvar(output_dir / "03_frame_t1.png", frame_t1)
    salvar(output_dir / "04_vetores_ida_volta.png", saida)
    salvar(output_dir / "05_oclusao.png", visual_oclusao)
    # Nome histórico citado no capítulo.
    salvar(output_dir / "03_vetores_lucas_kanade.png", saida)

    painel = mosaico(
        [
            rotular(frame_t, "Frame t"),
            rotular(pontos_vis, "Shi-Tomasi"),
            rotular(frame_t1, "Frame t+1"),
            rotular(saida, "Lucas-Kanade + ida-volta"),
            rotular(visual_oclusao, "Experimento com oclusão"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap09", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
