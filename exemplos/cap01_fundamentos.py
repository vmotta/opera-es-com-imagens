"""Capítulo 1: matrizes, pixels, BGR, ROI, canais, máscaras e bitwise.

Este laboratório usa somente imagens sintéticas para ser totalmente reprodutível.
Os resultados são gravados em outputs/cap01 por padrão.

Execute:
    python -m exemplos.cap01_fundamentos
"""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def demonstrar_view_e_copia() -> None:
    """Mostra no terminal a diferença entre uma view NumPy e uma cópia."""
    original = np.zeros((5, 5), dtype=np.uint8)

    view = original[1:4, 1:4]
    view[:] = 255

    print("\n--- VIEW NUMPY ---")
    print("Alterar a view também alterou a matriz original:")
    print(original)

    original = np.zeros((5, 5), dtype=np.uint8)
    copia = original[1:4, 1:4].copy()
    copia[:] = 255

    print("\n--- COPY NUMPY ---")
    print("Alterar a cópia não alterou a matriz original:")
    print(original)


def relatorio_imagem(nome: str, imagem: np.ndarray) -> None:
    """Imprime metadados importantes de uma imagem/array."""
    print(f"\n--- {nome.upper()} ---")
    print(f"shape: {imagem.shape}")
    print(f"dtype: {imagem.dtype}")
    print(f"size: {imagem.size}")
    print(f"ndim: {imagem.ndim}")

    altura, largura = imagem.shape[:2]
    print(f"altura: {altura}")
    print(f"largura: {largura}")
    print(f"pixels espaciais: {altura * largura}")

    if imagem.ndim == 3:
        print(f"canais: {imagem.shape[2]}")


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # ==========================================================================
    # 1. CRIAÇÃO DE UMA IMAGEM BGR SINTÉTICA
    # ==========================================================================
    # OpenCV/NumPy: (altura, largura, canais).
    # A tupla de preenchimento está em BGR, não RGB.
    imagem = np.full(
        (400, 600, 3),
        (180, 100, 50),
        dtype=np.uint8,
    )

    # Funções geométricas do OpenCV recebem pontos como (x, y).
    cv2.rectangle(
        imagem,
        (90, 70),
        (300, 330),
        (0, 210, 255),
        -1,
    )
    cv2.circle(
        imagem,
        (455, 205),
        82,
        (50, 255, 50),
        -1,
    )

    relatorio_imagem("imagem original", imagem)

    altura, largura, canais = imagem.shape
    centro_x = largura // 2
    centro_y = altura // 2

    # Acesso à matriz é [y, x] = [linha, coluna].
    pixel_central = imagem[centro_y, centro_x]
    print(
        f"Pixel central (x={centro_x}, y={centro_y}) "
        f"em BGR: {pixel_central.tolist()}"
    )

    # ==========================================================================
    # 2. ALTERAÇÃO VETORIZADA DE UMA REGIÃO
    # ==========================================================================
    modificada = imagem.copy()

    # Pinta um bloco 25x25 de vermelho.
    # BGR para vermelho puro = (0, 0, 255).
    modificada[10:35, 10:35] = (0, 0, 255)

    # ==========================================================================
    # 3. ROI: RECORTE, CÓPIA E INVERSÃO
    # ==========================================================================
    y1, y2 = 70, 270
    x1, x2 = 90, 300

    # .copy() evita que a edição da ROI altere automaticamente a região fonte.
    roi = modificada[y1:y2, x1:x2].copy()
    roi_invertida = cv2.bitwise_not(roi)

    print(
        "\nROI:",
        f"origem=(x={x1}, y={y1})",
        f"fim=(x={x2}, y={y2})",
        f"shape={roi.shape}",
    )

    # A ROI possui 200 px de altura e 210 px de largura.
    # A fatia de destino precisa ter exatamente o mesmo shape.
    destino_y1 = altura - roi.shape[0] - 10
    destino_x1 = largura - roi.shape[1] - 10
    destino_y2 = destino_y1 + roi.shape[0]
    destino_x2 = destino_x1 + roi.shape[1]

    modificada[
        destino_y1:destino_y2,
        destino_x1:destino_x2,
    ] = roi_invertida

    # ==========================================================================
    # 4. SEPARAÇÃO E RECOMPOSIÇÃO DOS CANAIS
    # ==========================================================================
    azul, verde, vermelho = cv2.split(modificada)

    print("\nShapes dos canais:")
    print("B:", azul.shape)
    print("G:", verde.shape)
    print("R:", vermelho.shape)

    # cv2.add faz adição saturada: resultados acima de 255 permanecem em 255.
    vermelho_forte = cv2.add(vermelho, 50)
    canais_modificados = cv2.merge((azul, verde, vermelho_forte))

    # Conversões explícitas entre representações.
    cinza = cv2.cvtColor(canais_modificados, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(canais_modificados, cv2.COLOR_BGR2RGB)
    bgr_reconstruido = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    print(
        "BGR -> RGB -> BGR preservou a matriz?",
        np.array_equal(canais_modificados, bgr_reconstruido),
    )

    # ==========================================================================
    # 5. MÁSCARAS BINÁRIAS
    # ==========================================================================
    mascara_circular = np.zeros((altura, largura), dtype=np.uint8)
    cv2.circle(
        mascara_circular,
        (centro_x, centro_y),
        135,
        255,
        -1,
    )

    mascara_retangular = np.zeros((altura, largura), dtype=np.uint8)
    cv2.rectangle(
        mascara_retangular,
        (180, 100),
        (500, 310),
        255,
        -1,
    )

    # Operações lógicas entre as duas máscaras.
    mascara_and = cv2.bitwise_and(
        mascara_circular,
        mascara_retangular,
    )
    mascara_or = cv2.bitwise_or(
        mascara_circular,
        mascara_retangular,
    )
    mascara_xor = cv2.bitwise_xor(
        mascara_circular,
        mascara_retangular,
    )
    mascara_not = cv2.bitwise_not(mascara_circular)

    # Aplica a máscara circular na imagem colorida.
    mascarada = cv2.bitwise_and(
        canais_modificados,
        canais_modificados,
        mask=mascara_circular,
    )

    # ==========================================================================
    # 6. DEMONSTRAÇÃO DE VIEW E COPY
    # ==========================================================================
    demonstrar_view_e_copia()

    # ==========================================================================
    # 7. EXPORTAÇÃO DOS RESULTADOS
    # ==========================================================================
    salvar(output_dir / "01_original.png", imagem)
    salvar(output_dir / "02_modificada_roi.png", modificada)
    salvar(output_dir / "03_canal_b.png", azul)
    salvar(output_dir / "04_canal_g.png", verde)
    salvar(output_dir / "05_canal_r.png", vermelho)
    salvar(output_dir / "06_vermelho_intensificado.png", canais_modificados)
    salvar(output_dir / "07_cinza.png", cinza)
    salvar(output_dir / "08_mascara_circular.png", mascara_circular)
    salvar(output_dir / "09_mascara_retangular.png", mascara_retangular)
    salvar(output_dir / "10_mascara_and.png", mascara_and)
    salvar(output_dir / "11_mascara_or.png", mascara_or)
    salvar(output_dir / "12_mascara_xor.png", mascara_xor)
    salvar(output_dir / "13_mascara_not.png", mascara_not)
    salvar(output_dir / "14_resultado_mascarado.png", mascarada)

    # Painel didático com os principais estágios.
    painel = mosaico(
        [
            rotular(imagem, "1. Original BGR"),
            rotular(modificada, "2. ROI invertida"),
            rotular(canais_modificados, "3. Vermelho + 50"),
            rotular(cinza, "4. Tons de cinza"),
            rotular(mascara_circular, "5. Mascara circular"),
            rotular(mascara_retangular, "6. Mascara retangular"),
            rotular(mascara_and, "7. AND"),
            rotular(mascara_or, "8. OR"),
            rotular(mascarada, "9. Imagem mascarada"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)

    print(
        f"\nConcluído. {canais} canais BGR processados. "
        f"Resultados em: {output_dir}"
    )


def main() -> None:
    parser = criar_parser("cap01", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
