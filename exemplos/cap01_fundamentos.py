"""Capítulo 1: matriz, pixels, ROI, canais e máscara binária."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # OpenCV organiza imagens coloridas como [altura, largura, canais BGR].
    imagem = np.full((400, 600, 3), (180, 100, 50), dtype=np.uint8)
    cv2.rectangle(imagem, (90, 70), (300, 330), (0, 210, 255), -1)
    cv2.circle(imagem, (455, 205), 82, (50, 255, 50), -1)

    altura, largura, canais = imagem.shape
    print(f"Forma={imagem.shape}; dtype={imagem.dtype}; elementos={imagem.size}")
    print(f"Pixel central BGR={imagem[altura // 2, largura // 2].tolist()}")

    modificada = imagem.copy()
    modificada[10:35, 10:35] = (0, 0, 255)

    # Fatiamento sempre usa [linhas y, colunas x]. A cópia evita alterar a fonte.
    y1, y2, x1, x2 = 70, 270, 90, 300
    roi = modificada[y1:y2, x1:x2].copy()
    roi_invertida = cv2.bitwise_not(roi)
    modificada[190:390, 380:590] = roi_invertida

    azul, verde, vermelho = cv2.split(modificada)
    # cv2.add satura em 255; uma soma uint8 comum poderia retornar ao zero.
    vermelho_forte = cv2.add(vermelho, 50)
    canais_modificados = cv2.merge((azul, verde, vermelho_forte))

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.circle(mascara, (largura // 2, altura // 2), 135, 255, -1)
    mascarada = cv2.bitwise_and(canais_modificados, canais_modificados, mask=mascara)

    salvar(output_dir / "01_original.png", imagem)
    salvar(output_dir / "02_roi_invertida.png", modificada)
    salvar(output_dir / "03_mascara.png", mascara)
    salvar(output_dir / "04_resultado_mascarado.png", mascarada)
    painel = mosaico(
        [
            rotular(imagem, "Original BGR"),
            rotular(modificada, "ROI copiada e invertida"),
            rotular(canais_modificados, "Canal vermelho + 50"),
            rotular(mascara, "Mascara binaria"),
            rotular(mascarada, "AND com mascara"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap01", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
