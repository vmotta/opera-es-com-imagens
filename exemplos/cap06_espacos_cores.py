"""Capítulo 6: HSV, Lab, canais e segmentação cromática com morfologia."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def destacar_maior_objeto(imagem: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    saida = imagem.copy()
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return saida
    maior = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(maior)
    x, y, w, h = cv2.boundingRect(maior)
    cv2.rectangle(saida, (x, y), (x + w, y + h), (0, 0, 0), 3)
    momentos = cv2.moments(maior)
    if momentos["m00"]:
        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"])
        cv2.circle(saida, (cx, cy), 7, (0, 0, 0), -1)
    print(f"Maior objeto segmentado: area={area:.1f}, bbox=({x},{y},{w},{h})")
    return saida


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    imagem = np.full((380, 680, 3), 235, dtype=np.uint8)
    cv2.circle(imagem, (135, 190), 88, (255, 30, 30), -1)  # azul BGR
    cv2.circle(imagem, (340, 190), 88, (30, 220, 30), -1)  # verde
    cv2.circle(imagem, (545, 190), 88, (20, 20, 235), -1)  # vermelho

    # Faixa escura atravessando parte da cena para demonstrar variação de brilho.
    sombra = imagem[:, 320:680].astype(np.float32) * 0.62
    imagem[:, 320:680] = np.clip(sombra, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    l_lab, a_lab, b_lab = cv2.split(lab)

    # Mostra numericamente o centro dos objetos.
    for nome, ponto in [("azul", (135, 190)), ("verde", (340, 190)), ("vermelho", (545, 190))]:
        x, y = ponto
        print(f"{nome}: BGR={imagem[y, x].tolist()} HSV={hsv[y, x].tolist()} Lab={lab[y, x].tolist()}")

    # --------------------------------------------------------------------------
    # 1. AZUL: uma faixa HSV.
    # --------------------------------------------------------------------------
    mascara_azul = cv2.inRange(
        hsv,
        np.array([100, 80, 40], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8),
    )

    # --------------------------------------------------------------------------
    # 2. VERMELHO: duas faixas por causa da circularidade de H.
    # --------------------------------------------------------------------------
    vermelho_baixo = cv2.inRange(
        hsv,
        np.array([0, 80, 35], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )
    vermelho_alto = cv2.inRange(
        hsv,
        np.array([170, 80, 35], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mascara_vermelha = cv2.bitwise_or(vermelho_baixo, vermelho_alto)

    # Pequenos defeitos sintéticos para que a morfologia tenha algo a corrigir.
    cv2.circle(mascara_azul, (40, 40), 4, 255, -1)
    cv2.circle(mascara_azul, (135, 190), 6, 0, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    azul_limpa = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel)
    azul_limpa = cv2.morphologyEx(azul_limpa, cv2.MORPH_CLOSE, kernel)
    vermelho_limpa = cv2.morphologyEx(mascara_vermelha, cv2.MORPH_OPEN, kernel)
    vermelho_limpa = cv2.morphologyEx(vermelho_limpa, cv2.MORPH_CLOSE, kernel)

    resultado_azul = cv2.bitwise_and(imagem, imagem, mask=azul_limpa)
    resultado_vermelho = cv2.bitwise_and(imagem, imagem, mask=vermelho_limpa)
    azul_medido = destacar_maior_objeto(imagem, azul_limpa)

    resultados = {
        "01_original_bgr.png": imagem,
        "02_canal_h.png": h,
        "03_canal_s.png": s,
        "04_canal_v.png": v,
        "05_lab_l.png": l_lab,
        "06_lab_a.png": a_lab,
        "07_lab_b.png": b_lab,
        "08_mascara_azul_bruta.png": mascara_azul,
        "09_mascara_azul_limpa.png": azul_limpa,
        "10_resultado_azul.png": resultado_azul,
        "11_mascara_vermelha_dupla.png": mascara_vermelha,
        "12_resultado_vermelho.png": resultado_vermelho,
        "13_azul_bbox_centroide.png": azul_medido,
    }
    for nome, resultado in resultados.items():
        salvar(output_dir / nome, resultado)

    painel = mosaico(
        [
            rotular(imagem, "Cena BGR com sombra"),
            rotular(h, "H: matiz"),
            rotular(s, "S: saturacao"),
            rotular(v, "V: brilho"),
            rotular(mascara_azul, "Azul: mascara bruta"),
            rotular(azul_limpa, "Azul: morfologia"),
            rotular(resultado_azul, "Azul isolado"),
            rotular(mascara_vermelha, "Vermelho: 2 faixas"),
            rotular(resultado_vermelho, "Vermelho isolado"),
            rotular(azul_medido, "Maior azul medido"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap06", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
