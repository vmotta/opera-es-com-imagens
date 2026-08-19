"""Capítulo 14: criação, pré-processamento e OCR opcional com Tesseract."""

from __future__ import annotations

import shutil

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.full((260, 900, 3), 205, dtype=np.uint8)
    cv2.putText(
        imagem,
        "TOTAL A PAGAR: R$ 1.450,00",
        (38, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (42, 42, 42),
        3,
        cv2.LINE_AA,
    )
    gerador = np.random.default_rng(14)
    iluminacao = np.tile(np.linspace(0, 35, imagem.shape[1], dtype=np.uint8), (imagem.shape[0], 1))
    ruidosa = cv2.add(imagem, cv2.merge((iluminacao, iluminacao, iluminacao)))
    impulsos = gerador.random(imagem.shape[:2])
    ruidosa[impulsos < 0.006] = (65, 65, 65)
    ruidosa[impulsos > 0.994] = (255, 255, 255)

    cinza = cv2.cvtColor(ruidosa, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    salvar(output_dir / "01_documento_ruidoso.png", ruidosa)
    salvar(output_dir / "02_pre_processado.png", binaria)

    if shutil.which("tesseract") is None:
        print("[AVISO] Tesseract não está no PATH. As imagens foram preparadas, mas o OCR foi omitido.")
        return
    try:
        import pytesseract
    except ImportError:
        print('[AVISO] Instale o extra de OCR com: python -m pip install -e ".[ocr]"')
        return

    idiomas = set(pytesseract.get_languages(config=""))
    idioma = "por" if "por" in idiomas else "eng"
    configuracao = f"--oem 3 --psm 6 -l {idioma}"
    texto = pytesseract.image_to_string(binaria, config=configuracao).strip()
    print(f"Texto extraído ({idioma}): {texto!r}")


def main() -> None:
    parser = criar_parser("cap14", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
