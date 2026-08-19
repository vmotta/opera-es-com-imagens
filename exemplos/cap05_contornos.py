"""Capítulo 5: contornos, hierarquia, área, perímetro, caixas, centroide e forma."""

from __future__ import annotations

import math

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def classificar_forma(contorno: np.ndarray) -> tuple[str, np.ndarray, float]:
    """Classificador geométrico didático por vértices e circularidade."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    aproximado = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
    circularidade = 4 * math.pi * area / (perimetro**2) if perimetro > 0 else 0.0

    vertices = len(aproximado)
    if vertices == 3:
        classe = "triangulo"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(aproximado)
        razao = w / h if h else 0.0
        classe = "quadrado" if 0.90 <= razao <= 1.10 else "retangulo"
    elif circularidade >= 0.80:
        classe = "circulo"
    else:
        classe = f"poligono({vertices})"
    return classe, aproximado, circularidade


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    imagem = np.zeros((540, 700, 3), dtype=np.uint8)
    cv2.rectangle(imagem, (45, 50), (235, 170), (255, 255, 255), -1)
    cv2.circle(imagem, (430, 125), 72, (255, 255, 255), -1)
    triangulo = np.array([[105, 315], [270, 480], [35, 480]], dtype=np.int32)
    cv2.fillPoly(imagem, [triangulo], (255, 255, 255))

    # Retângulo rotacionado: ajuda a comparar boundingRect e minAreaRect.
    ret_rot = ((500, 390), (180, 80), 28)
    caixa_rot = cv2.boxPoints(ret_rot).astype(np.int32)
    cv2.fillPoly(imagem, [caixa_rot], (255, 255, 255))

    # Anel para demonstrar hierarquia pai-filho.
    cv2.circle(imagem, (625, 120), 48, (255, 255, 255), -1)
    cv2.circle(imagem, (625, 120), 22, (0, 0, 0), -1)

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(cinza, 127, 255, cv2.THRESH_BINARY)

    # --------------------------------------------------------------------------
    # 1. RETR_EXTERNAL E RETR_TREE: compare quantos contornos aparecem.
    # --------------------------------------------------------------------------
    contornos_ext, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_tree, hierarquia = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"RETR_EXTERNAL: {len(contornos_ext)} contornos")
    print(f"RETR_TREE: {len(contornos_tree)} contornos")
    if hierarquia is not None:
        print("Hierarquia [next, prev, child, parent]:")
        for i, dados in enumerate(hierarquia[0]):
            print(f"  contorno {i}: {dados.tolist()}")

    resultado = imagem.copy()
    medidas: list[tuple[float, np.ndarray]] = []
    for contorno in contornos_ext:
        area = cv2.contourArea(contorno)
        if area >= 500:
            medidas.append((area, contorno))

    # A ordem original não possui significado semântico; x fornece IDs estáveis.
    medidas.sort(key=lambda item: cv2.boundingRect(item[1])[0])

    for indice, (area, contorno) in enumerate(medidas, start=1):
        perimetro = cv2.arcLength(contorno, True)
        x, y, largura, altura = cv2.boundingRect(contorno)
        momentos = cv2.moments(contorno)
        cx = int(momentos["m10"] / momentos["m00"]) if momentos["m00"] else x
        cy = int(momentos["m01"] / momentos["m00"]) if momentos["m00"] else y
        classe, aproximado, circularidade = classificar_forma(contorno)
        razao = largura / altura if altura else 0.0

        # Caixa alinhada aos eixos.
        cv2.rectangle(resultado, (x, y), (x + largura, y + altura), (0, 170, 0), 2)

        # Caixa rotacionada mínima.
        min_rect = cv2.minAreaRect(contorno)
        min_box = cv2.boxPoints(min_rect).astype(np.int32)
        cv2.polylines(resultado, [min_box], True, (255, 120, 0), 2)

        # Polígono aproximado e centroide.
        cv2.polylines(resultado, [aproximado], True, (0, 220, 255), 2)
        cv2.circle(resultado, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            resultado,
            f"{indice}:{classe}",
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
        )
        print(
            f"Objeto {indice}: classe={classe}, centro=({cx},{cy}), "
            f"area={area:.1f}, perimetro={perimetro:.1f}, "
            f"circularidade={circularidade:.3f}, razao={razao:.3f}, "
            f"vertices={len(aproximado)}"
        )

    # Visualização separada da hierarquia: todos os contornos, incluindo o buraco.
    visual_tree = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(visual_tree, contornos_tree, -1, (0, 255, 0), 2)
    for i, contorno in enumerate(contornos_tree):
        x, y, _, _ = cv2.boundingRect(contorno)
        cv2.putText(
            visual_tree,
            str(i),
            (x, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    salvar(output_dir / "01_binaria.png", binaria)
    salvar(output_dir / "02_contornos_medidos.png", resultado)
    salvar(output_dir / "03_hierarquia_retr_tree.png", visual_tree)

    painel = mosaico(
        [
            rotular(imagem, "Formas originais"),
            rotular(resultado, "Medidas e caixas"),
            rotular(visual_tree, "RETR_TREE e buraco"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap05", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
