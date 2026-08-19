"""Capítulo 7: correspondência de modelo e classificador Haar em cascata."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)

    cena = np.full((340, 500, 3), 155, dtype=np.uint8)
    cv2.circle(cena, (55, 55), 22, (0, 0, 255), -1)
    cv2.rectangle(cena, (105, 235), (150, 280), (255, 0, 0), -1)
    cv2.circle(cena, (360, 170), 36, (0, 255, 255), -1)

    template = np.full((72, 72, 3), 155, dtype=np.uint8)
    cv2.circle(template, (36, 36), 36, (0, 255, 255), -1)
    mapa_similaridade = cv2.matchTemplate(cena, template, cv2.TM_CCOEFF_NORMED)
    _, melhor_valor, _, melhor_local = cv2.minMaxLoc(mapa_similaridade)
    print(f"Melhor similaridade do template: {melhor_valor:.3f}")

    marcado = cena.copy()
    altura_template, largura_template = template.shape[:2]
    if melhor_valor >= 0.80:
        x, y = melhor_local
        cv2.rectangle(
            marcado,
            (x, y),
            (x + largura_template, y + altura_template),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            marcado,
            f"alvo {melhor_valor:.2f}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

    # O Haar foi treinado com fotografias; o desenho abaixo serve para observar
    # que uma forma "parecida com rosto" não garante uma detecção.
    rosto_didatico = np.full((420, 420, 3), 245, dtype=np.uint8)
    cv2.ellipse(rosto_didatico, (210, 215), (120, 155), 0, 0, 360, (150, 190, 225), -1)
    cv2.circle(rosto_didatico, (165, 185), 14, (30, 30, 30), -1)
    cv2.circle(rosto_didatico, (255, 185), 14, (30, 30, 30), -1)
    cv2.ellipse(rosto_didatico, (210, 265), (50, 24), 0, 0, 180, (30, 30, 30), 5)

    caminho_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascata = cv2.CascadeClassifier(caminho_xml)
    if cascata.empty():
        raise RuntimeError(f"Classificador Haar não carregado: {caminho_xml}")
    cinza = cv2.cvtColor(rosto_didatico, cv2.COLOR_BGR2GRAY)
    rostos = cascata.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for x, y, largura, altura in rostos:
        cv2.rectangle(rosto_didatico, (x, y), (x + largura, y + altura), (0, 255, 0), 3)
    cv2.putText(
        rosto_didatico,
        f"deteccoes Haar: {len(rostos)}",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 180),
        2,
    )
    print(f"Detecções Haar na ilustração: {len(rostos)}")

    salvar(output_dir / "01_template.png", template)
    salvar(output_dir / "02_template_encontrado.png", marcado)
    salvar(output_dir / "03_haar_experimento.png", rosto_didatico)


def main() -> None:
    parser = criar_parser("cap07", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
