"""Capítulo 11: caixas YOLO, IoU, NMS manual, OpenCV e NMS por classe."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar

ROTULOS = ["pessoa", "bicicleta", "carro"]
CORES = [(30, 210, 255), (255, 140, 30), (60, 220, 60)]


def xywh_para_xyxy(caixa):
    x, y, w, h = caixa
    return [x, y, x + w, y + h]


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    uniao = area_a + area_b - inter
    return 0.0 if uniao <= 0 else float(inter / uniao)


def nms_manual(indices, caixas, confiancas, limiar_iou):
    """NMS didático para um grupo que já representa a mesma classe."""
    ordem = sorted(indices, key=lambda i: confiancas[i], reverse=True)
    manter = []
    while ordem:
        atual = ordem.pop(0)
        manter.append(atual)
        caixa_atual = xywh_para_xyxy(caixas[atual])
        ordem = [
            j for j in ordem
            if iou_xyxy(caixa_atual, xywh_para_xyxy(caixas[j])) <= limiar_iou
        ]
    return manter


def desenhar(imagem, caixas, confiancas, classes, indices=None, titulo=None):
    saida = imagem.copy()
    selecionados = range(len(caixas)) if indices is None else indices
    for indice in selecionados:
        x, y, w, h = caixas[indice]
        classe = classes[indice]
        cor = CORES[classe]
        cv2.rectangle(saida, (x, y), (x + w, y + h), cor, 3)
        cv2.putText(
            saida,
            f"{ROTULOS[classe]} {confiancas[indice]:.2f}",
            (x, max(25, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            cor,
            2,
        )
    if titulo:
        cv2.putText(saida, titulo, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2)
    return saida


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.full((600, 800, 3), 105, dtype=np.uint8)
    cv2.rectangle(imagem, (150, 200), (250, 505), (155, 155, 155), -1)
    cv2.rectangle(imagem, (215, 330), (330, 505), (95, 95, 160), -1)  # segunda pessoa próxima
    cv2.rectangle(imagem, (400, 300), (700, 455), (200, 60, 50), -1)

    h, w = imagem.shape[:2]
    # cx, cy, largura, altura, confiança, classe — coordenadas normalizadas.
    predicoes = [
        [550 / w, 377 / h, 300 / w, 155 / h, 0.95, 2],
        [540 / w, 372 / h, 292 / w, 146 / h, 0.82, 2],
        [200 / w, 352 / h, 100 / w, 305 / h, 0.88, 0],
        [205 / w, 357 / h, 108 / w, 312 / h, 0.70, 0],
        [272 / w, 417 / h, 116 / w, 178 / h, 0.86, 0],
        # Bicicleta sobreposta à pessoa: deve sobreviver ao NMS por classe.
        [220 / w, 420 / h, 210 / w, 120 / h, 0.78, 1],
    ]

    caixas: list[list[int]] = []
    confiancas: list[float] = []
    classes: list[int] = []
    for cx, cy, largura, altura, confianca, classe in predicoes:
        largura_px, altura_px = int(largura * w), int(altura * h)
        x = int(cx * w - largura_px / 2)
        y = int(cy * h - altura_px / 2)
        caixas.append([x, y, largura_px, altura_px])
        confiancas.append(float(confianca))
        classes.append(int(classe))

    # --------------------------------------------------------------------------
    # 1. IoU ENTRE ALGUMAS CAIXAS PARA DAR SIGNIFICADO NUMÉRICO À SOBREPOSIÇÃO.
    # --------------------------------------------------------------------------
    print("IoU carro duplicado:", f"{iou_xyxy(xywh_para_xyxy(caixas[0]), xywh_para_xyxy(caixas[1])):.3f}")
    print("IoU pessoa duplicada:", f"{iou_xyxy(xywh_para_xyxy(caixas[2]), xywh_para_xyxy(caixas[3])):.3f}")
    print("IoU pessoas distintas:", f"{iou_xyxy(xywh_para_xyxy(caixas[2]), xywh_para_xyxy(caixas[4])):.3f}")
    print("IoU pessoa/bicicleta:", f"{iou_xyxy(xywh_para_xyxy(caixas[2]), xywh_para_xyxy(caixas[5])):.3f}")

    antes = desenhar(imagem, caixas, confiancas, classes, titulo="Todas as hipóteses")

    # --------------------------------------------------------------------------
    # 2. NMS MANUAL POR CLASSE EM VÁRIOS LIMIARES.
    # --------------------------------------------------------------------------
    resultados_limiar = {}
    for limiar_iou in (0.15, 0.35, 0.70):
        sobreviventes = []
        for classe in sorted(set(classes)):
            indices_classe = [i for i, valor in enumerate(classes) if valor == classe and confiancas[i] >= 0.5]
            sobreviventes.extend(nms_manual(indices_classe, caixas, confiancas, limiar_iou))
        sobreviventes.sort()
        print(f"NMS manual IoU={limiar_iou:.2f}: sobreviventes={sobreviventes}")
        resultados_limiar[limiar_iou] = desenhar(
            imagem,
            caixas,
            confiancas,
            classes,
            sobreviventes,
            titulo=f"NMS IoU={limiar_iou:.2f}",
        )

    # --------------------------------------------------------------------------
    # 3. NMS DO OPENCV POR CLASSE PARA CONFERIR A IMPLEMENTAÇÃO DIDÁTICA.
    # --------------------------------------------------------------------------
    sobreviventes_cv: list[int] = []
    for classe in sorted(set(classes)):
        globais = [i for i, valor in enumerate(classes) if valor == classe]
        caixas_classe = [caixas[i] for i in globais]
        confiancas_classe = [confiancas[i] for i in globais]
        locais = cv2.dnn.NMSBoxes(caixas_classe, confiancas_classe, 0.5, 0.35)
        sobreviventes_cv.extend(globais[int(i)] for i in np.array(locais).reshape(-1))
    sobreviventes_cv.sort()
    print("OpenCV NMSBoxes IoU=.35:", sobreviventes_cv)
    depois = desenhar(imagem, caixas, confiancas, classes, sobreviventes_cv, titulo="OpenCV NMS por classe")

    # Comparação proposital: NMS global pode suprimir classes diferentes.
    globais_nms = cv2.dnn.NMSBoxes(caixas, confiancas, 0.5, 0.35)
    globais_idx = [int(i) for i in np.array(globais_nms).reshape(-1)]
    global_vis = desenhar(imagem, caixas, confiancas, classes, globais_idx, titulo="NMS global: cuidado")
    print("NMS global (sem separar classe):", globais_idx)

    salvar(output_dir / "01_antes_nms.png", antes)
    salvar(output_dir / "02_depois_nms.png", depois)
    salvar(output_dir / "03_nms_iou_015.png", resultados_limiar[0.15])
    salvar(output_dir / "04_nms_iou_035.png", resultados_limiar[0.35])
    salvar(output_dir / "05_nms_iou_070.png", resultados_limiar[0.70])
    salvar(output_dir / "06_nms_global.png", global_vis)

    painel = mosaico(
        [
            rotular(antes, "Antes do NMS"),
            rotular(resultados_limiar[0.15], "NMS 0.15"),
            rotular(resultados_limiar[0.35], "NMS 0.35"),
            rotular(resultados_limiar[0.70], "NMS 0.70"),
            rotular(depois, "OpenCV por classe"),
            rotular(global_vis, "OpenCV global"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap11", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
