"""Capítulo 14: pré-processamento, deskew, OCR opcional, confiança e validação textual."""

from __future__ import annotations

import re
import shutil

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def rotacionar(imagem: np.ndarray, angulo: float, valor_borda=255) -> np.ndarray:
    altura, largura = imagem.shape[:2]
    matriz = cv2.getRotationMatrix2D((largura / 2, altura / 2), angulo, 1.0)
    return cv2.warpAffine(
        imagem,
        matriz,
        (largura, altura),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=valor_borda,
    )


def estimar_skew(binaria_texto_preto: np.ndarray) -> float:
    """Estima inclinação por minAreaRect em pixels escuros; valor didático."""
    coords_yx = np.column_stack(np.where(binaria_texto_preto < 128))
    if len(coords_yx) < 10:
        return 0.0
    # minAreaRect espera pontos (x, y), por isso invertemos as colunas.
    coords_xy = coords_yx[:, ::-1].astype(np.float32)
    angulo = float(cv2.minAreaRect(coords_xy)[-1])
    # Convenção típica do OpenCV para o retângulo mínimo.
    if angulo > 45:
        angulo -= 90
    return angulo


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    # --------------------------------------------------------------------------
    # 1. DOCUMENTO SINTÉTICO COM DUAS LINHAS, GRADIENTE E RUÍDO.
    # --------------------------------------------------------------------------
    documento = np.full((360, 960, 3), 215, dtype=np.uint8)
    cv2.putText(documento, "RECIBO DIDATICO", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (38, 38, 38), 3, cv2.LINE_AA)
    cv2.putText(documento, "TOTAL A PAGAR: R$ 1.450,00", (45, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (42, 42, 42), 3, cv2.LINE_AA)
    cv2.putText(documento, "DATA: 19/08/2026", (45, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (42, 42, 42), 3, cv2.LINE_AA)

    iluminacao = np.tile(np.linspace(0, 38, documento.shape[1], dtype=np.uint8), (documento.shape[0], 1))
    ruidosa = cv2.add(documento, cv2.merge((iluminacao, iluminacao, iluminacao)))
    rng = np.random.default_rng(14)
    impulsos = rng.random(ruidosa.shape[:2])
    ruidosa[impulsos < 0.004] = (65, 65, 65)
    ruidosa[impulsos > 0.996] = (255, 255, 255)

    # Inclinação proposital.
    inclinada = rotacionar(ruidosa, 7.0, valor_borda=(255, 255, 255))
    cinza = cv2.cvtColor(inclinada, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------------------------
    # 2. COMPARA OTSU E LIMIAR ADAPTATIVO.
    # --------------------------------------------------------------------------
    suave = cv2.GaussianBlur(cinza, (3, 3), 0)
    valor_otsu, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adaptativa = cv2.adaptiveThreshold(
        cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    print(f"Limiar Otsu escolhido: {valor_otsu:.1f}")

    # --------------------------------------------------------------------------
    # 3. DESKEW DA BINÁRIA OTSU.
    # --------------------------------------------------------------------------
    angulo_estimado = estimar_skew(otsu)
    corrigida = rotacionar(otsu, angulo_estimado, valor_borda=255)
    print(f"Ângulo estimado pelo minAreaRect: {angulo_estimado:.2f}°; correção aplicada no mesmo sinal")

    # Uma segunda versão com ampliação pode ajudar caracteres pequenos.
    ampliada = cv2.resize(corrigida, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    salvar(output_dir / "01_documento_ruidoso.png", inclinada)
    salvar(output_dir / "02_pre_processado.png", corrigida)
    salvar(output_dir / "03_otsu_inclinado.png", otsu)
    salvar(output_dir / "04_adaptativo.png", adaptativa)
    salvar(output_dir / "05_deskew.png", corrigida)
    salvar(output_dir / "06_deskew_ampliado.png", ampliada)

    painel = mosaico(
        [
            rotular(inclinada, "Documento inclinado"),
            rotular(otsu, "Otsu"),
            rotular(adaptativa, "Adaptativo"),
            rotular(corrigida, "Deskew"),
            rotular(ampliada, "Deskew + ampliação"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)

    # --------------------------------------------------------------------------
    # 4. OCR OPCIONAL: o laboratório continua útil mesmo sem Tesseract.
    # --------------------------------------------------------------------------
    if shutil.which("tesseract") is None:
        print("[AVISO] Tesseract não está no PATH. Pré-processamento concluído; OCR opcional foi omitido.")
        return
    try:
        import pytesseract
    except ImportError:
        print('[AVISO] Instale o extra de OCR com: python -m pip install -e ".[ocr]"')
        return

    idiomas = set(pytesseract.get_languages(config=""))
    idioma = "por" if "por" in idiomas else "eng" if "eng" in idiomas else None
    if idioma is None:
        print("[AVISO] Nenhum idioma por/eng disponível no Tesseract.")
        return

    # Compara hipóteses de layout.
    for psm in (6, 11):
        config = f"--oem 3 --psm {psm} -l {idioma}"
        texto = pytesseract.image_to_string(ampliada, config=config).strip()
        print(f"\nOCR PSM {psm} ({idioma}):\n{texto}")

    config = f"--oem 3 --psm 6 -l {idioma}"
    texto = pytesseract.image_to_string(ampliada, config=config).strip()
    dados = pytesseract.image_to_data(
        ampliada,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    confidencias = []
    visual_palavras = cv2.cvtColor(ampliada, cv2.COLOR_GRAY2BGR)
    for i, palavra in enumerate(dados["text"]):
        palavra = palavra.strip()
        try:
            conf = float(dados["conf"][i])
        except (TypeError, ValueError):
            continue
        if palavra and conf >= 0:
            confidencias.append(conf)
        if palavra and conf >= 60:
            x, y = dados["left"][i], dados["top"][i]
            w, h = dados["width"][i], dados["height"][i]
            cv2.rectangle(visual_palavras, (x, y), (x + w, y + h), (0, 180, 0), 2)
            cv2.putText(visual_palavras, f"{conf:.0f}", (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)

    if confidencias:
        print(f"Confiança média das palavras retornadas: {np.mean(confidencias):.1f}")
    salvar(output_dir / "07_palavras_confianca.png", visual_palavras)

    # Extração estruturada: regex é seguida por validação mínima.
    padrao_valor = r"R\$\s*([0-9][0-9\.]*[,\.][0-9]{2})"
    encontrado = re.search(padrao_valor, texto, flags=re.IGNORECASE)
    if encontrado:
        valor_texto = encontrado.group(1)
        normalizado = valor_texto.replace(".", "").replace(",", ".")
        try:
            valor = float(normalizado)
            print(f"Valor monetário extraído/normalizado: {valor:.2f}")
        except ValueError:
            print("Valor encontrado pela regex, mas falhou na validação numérica:", valor_texto)
    else:
        print("Regex não encontrou um valor monetário confiável na saída OCR.")


def main() -> None:
    parser = criar_parser("cap14", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
