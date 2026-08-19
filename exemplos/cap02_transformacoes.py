"""Capítulo 2: escala, interpolação, translação, rotação, afim e perspectiva."""

from __future__ import annotations

import math

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def rotacionar_sem_corte(imagem: np.ndarray, angulo: float) -> np.ndarray:
    """Rotaciona a imagem aumentando o canvas para preservar os quatro cantos."""
    altura, largura = imagem.shape[:2]
    centro = (largura / 2, altura / 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    cos = abs(matriz[0, 0])
    sin = abs(matriz[0, 1])
    nova_largura = int(math.ceil(altura * sin + largura * cos))
    nova_altura = int(math.ceil(altura * cos + largura * sin))

    matriz[0, 2] += nova_largura / 2 - largura / 2
    matriz[1, 2] += nova_altura / 2 - altura / 2

    return cv2.warpAffine(
        imagem,
        matriz,
        (nova_largura, nova_altura),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(225, 225, 225),
    )


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # Cena propositalmente não quadrada: ajuda a perceber trocas entre largura e altura.
    imagem = np.full((320, 480, 3), 225, dtype=np.uint8)
    cv2.rectangle(imagem, (55, 45), (315, 275), (0, 105, 205), -1)
    cv2.circle(imagem, (195, 160), 62, (210, 55, 40), -1)
    cv2.putText(imagem, "OpenCV", (115, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(imagem, (350, 45), (440, 135), (30, 30, 30), 5)
    cv2.putText(imagem, "x", (425, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    altura, largura = imagem.shape[:2]

    print(f"Imagem original: largura={largura}, altura={altura}")

    # --------------------------------------------------------------------------
    # 1. INTERPOLAÇÃO: mesma ampliação com métodos diferentes.
    # --------------------------------------------------------------------------
    tamanho_ampliado = (largura * 2, altura * 2)
    vizinho = cv2.resize(imagem, tamanho_ampliado, interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(imagem, tamanho_ampliado, interpolation=cv2.INTER_LINEAR)
    cubica = cv2.resize(imagem, tamanho_ampliado, interpolation=cv2.INTER_CUBIC)
    reduzida = cv2.resize(imagem, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # --------------------------------------------------------------------------
    # 2. MÁSCARA DE CLASSES: mostra por que INTER_NEAREST preserva rótulos.
    # --------------------------------------------------------------------------
    mascara_classes = np.zeros((80, 120), dtype=np.uint8)
    mascara_classes[:, 40:80] = 1
    mascara_classes[:, 80:] = 2
    mascara_nearest = cv2.resize(mascara_classes, (360, 240), interpolation=cv2.INTER_NEAREST)
    mascara_linear = cv2.resize(mascara_classes, (360, 240), interpolation=cv2.INTER_LINEAR)
    print("Valores únicos máscara NEAREST:", np.unique(mascara_nearest).tolist())
    print("Valores únicos máscara LINEAR:", np.unique(mascara_linear).tolist())

    # Para visualizar classes 0, 1 e 2 como tons distintos.
    mascara_nearest_vis = (mascara_nearest * 120).astype(np.uint8)
    mascara_linear_vis = np.clip(mascara_linear * 120, 0, 255).astype(np.uint8)

    # --------------------------------------------------------------------------
    # 3. TRANSLAÇÃO.
    # --------------------------------------------------------------------------
    matriz_translacao = np.float32([[1, 0, 70], [0, 1, 45]])
    transladada = cv2.warpAffine(
        imagem,
        matriz_translacao,
        (largura, altura),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )

    # --------------------------------------------------------------------------
    # 4. ROTAÇÃO: canvas fixo versus canvas expandido.
    # --------------------------------------------------------------------------
    matriz_rotacao = cv2.getRotationMatrix2D((largura / 2, altura / 2), 35, 1.0)
    rotacionada_cortada = cv2.warpAffine(
        imagem,
        matriz_rotacao,
        (largura, altura),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(225, 225, 225),
    )
    rotacionada_sem_corte = rotacionar_sem_corte(imagem, 35)

    # --------------------------------------------------------------------------
    # 5. TRANSFORMAÇÃO AFIM A PARTIR DE TRÊS PARES DE PONTOS.
    # --------------------------------------------------------------------------
    origem_afim = np.float32([[55, 45], [315, 45], [55, 275]])
    destino_afim = np.float32([[25, 90], [340, 40], [105, 295]])
    matriz_afim = cv2.getAffineTransform(origem_afim, destino_afim)
    afim = cv2.warpAffine(imagem, matriz_afim, (largura, altura), borderValue=(225, 225, 225))

    # --------------------------------------------------------------------------
    # 6. PERSPECTIVA / HOMOGRAFIA A PARTIR DE QUATRO PARES.
    # --------------------------------------------------------------------------
    origem_perspectiva = np.float32([[55, 45], [315, 45], [55, 275], [315, 275]])
    destino_perspectiva = np.float32([[20, 35], [370, 10], [75, 305], [325, 270]])
    matriz_perspectiva = cv2.getPerspectiveTransform(origem_perspectiva, destino_perspectiva)
    perspectiva = cv2.warpPerspective(
        imagem,
        matriz_perspectiva,
        (largura, altura),
        borderValue=(225, 225, 225),
    )

    resultados = {
        "01_original.png": imagem,
        "02_ampliada_nearest.png": vizinho,
        "03_ampliada_linear.png": linear,
        "04_ampliada_cubica.png": cubica,
        "05_reduzida_area.png": reduzida,
        "06_mascara_nearest.png": mascara_nearest_vis,
        "07_mascara_linear.png": mascara_linear_vis,
        "08_transladada.png": transladada,
        "09_rotacionada_canvas_fixo.png": rotacionada_cortada,
        "10_rotacionada_sem_corte.png": rotacionada_sem_corte,
        "11_afim.png": afim,
        "12_perspectiva.png": perspectiva,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)

    painel = mosaico(
        [
            rotular(imagem, "Original"),
            rotular(reduzida, "Reducao INTER_AREA"),
            rotular(transladada, "Translacao"),
            rotular(rotacionada_cortada, "Rotacao: canvas fixo"),
            rotular(rotacionada_sem_corte, "Rotacao sem corte"),
            rotular(afim, "Transformacao afim"),
            rotular(perspectiva, "Perspectiva"),
            rotular(mascara_nearest_vis, "Rotulos: NEAREST"),
            rotular(mascara_linear_vis, "Rotulos: LINEAR"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap02", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
