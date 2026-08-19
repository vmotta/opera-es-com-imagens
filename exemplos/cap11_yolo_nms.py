"""Capítulo 11: caixas YOLO normalizadas, IoU e supressão não máxima."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar

ROTULOS = ["pessoa", "bicicleta", "carro"]
CORES = [(30, 210, 255), (255, 140, 30), (60, 220, 60)]


def desenhar(imagem, caixas, confiancas, classes, indices=None):
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
            0.65,
            cor,
            2,
        )
    return saida


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.full((600, 800, 3), 105, dtype=np.uint8)
    cv2.rectangle(imagem, (150, 200), (250, 505), (155, 155, 155), -1)
    cv2.rectangle(imagem, (400, 300), (700, 455), (200, 60, 50), -1)

    h, w = imagem.shape[:2]
    # cx, cy, largura, altura, confiança, classe; coordenadas relativas a 0..1.
    predicoes = [
        [550 / w, 377 / h, 300 / w, 155 / h, 0.95, 2],
        [540 / w, 372 / h, 292 / w, 146 / h, 0.82, 2],
        [200 / w, 352 / h, 100 / w, 305 / h, 0.88, 0],
        [205 / w, 357 / h, 108 / w, 312 / h, 0.70, 0],
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

    antes = desenhar(imagem, caixas, confiancas, classes)
    # NMS deve ser aplicado por classe para que objetos distintos não se eliminem.
    sobreviventes: list[int] = []
    for classe in sorted(set(classes)):
        globais = [i for i, valor in enumerate(classes) if valor == classe]
        caixas_classe = [caixas[i] for i in globais]
        confiancas_classe = [confiancas[i] for i in globais]
        locais = cv2.dnn.NMSBoxes(caixas_classe, confiancas_classe, 0.5, 0.35)
        sobreviventes.extend(globais[int(i)] for i in np.array(locais).reshape(-1))
    sobreviventes.sort()
    print(f"Caixas antes={len(caixas)}; depois do NMS={len(sobreviventes)}")
    depois = desenhar(imagem, caixas, confiancas, classes, sobreviventes)

    salvar(output_dir / "01_antes_nms.png", antes)
    salvar(output_dir / "02_depois_nms.png", depois)


def main() -> None:
    parser = criar_parser("cap11", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
