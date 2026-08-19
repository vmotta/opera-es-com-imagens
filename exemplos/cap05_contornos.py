"""Capítulo 5: contornos, área, perímetro, centroide e bounding box."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    imagem = np.zeros((500, 560, 3), dtype=np.uint8)
    cv2.rectangle(imagem, (45, 50), (220, 165), (255, 255, 255), -1)
    cv2.circle(imagem, (410, 125), 72, (255, 255, 255), -1)
    triangulo = np.array([[105, 305], [270, 460], [40, 460]], dtype=np.int32)
    cv2.fillPoly(imagem, [triangulo], (255, 255, 255))

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(cinza, 127, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultado = imagem.copy()
    medidas: list[tuple[float, np.ndarray]] = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area >= 500:
            medidas.append((area, contorno))

    # A ordem original depende do algoritmo. Ordenar por x torna o resultado reproduzível.
    medidas.sort(key=lambda item: cv2.boundingRect(item[1])[0])
    for indice, (area, contorno) in enumerate(medidas, start=1):
        perimetro = cv2.arcLength(contorno, True)
        x, y, largura, altura = cv2.boundingRect(contorno)
        momentos = cv2.moments(contorno)
        cx = int(momentos["m10"] / momentos["m00"]) if momentos["m00"] else x
        cy = int(momentos["m01"] / momentos["m00"]) if momentos["m00"] else y
        circularidade = 4 * np.pi * area / (perimetro**2) if perimetro else 0

        cv2.rectangle(resultado, (x, y), (x + largura, y + altura), (0, 255, 0), 2)
        cv2.circle(resultado, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            resultado,
            f"Obj {indice}",
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
        )
        print(
            f"Objeto {indice}: centro=({cx},{cy}), área={area:.1f}, "
            f"perímetro={perimetro:.1f}, circularidade={circularidade:.3f}"
        )

    salvar(output_dir / "01_binaria.png", binaria)
    salvar(output_dir / "02_contornos_medidos.png", resultado)


def main() -> None:
    parser = criar_parser("cap05", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
