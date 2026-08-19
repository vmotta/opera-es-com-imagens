"""Capítulo 18: responsabilidade celular e inferência opcional com Ultralytics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersecao = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    uniao = area_a + area_b - intersecao
    return intersecao / uniao if uniao else 0.0


def diagrama_grid() -> np.ndarray:
    tamanho, celulas = 700, 7
    tela = np.full((tamanho, tamanho, 3), 245, dtype=np.uint8)
    passo = tamanho // celulas
    for i in range(celulas + 1):
        cv2.line(tela, (i * passo, 0), (i * passo, tamanho), (165, 165, 165), 1)
        cv2.line(tela, (0, i * passo), (tamanho, i * passo), (165, 165, 165), 1)

    caixa = (240, 190, 555, 505)
    cx, cy = (caixa[0] + caixa[2]) // 2, (caixa[1] + caixa[3]) // 2
    coluna, linha = cx // passo, cy // passo
    cv2.rectangle(
        tela,
        (coluna * passo, linha * passo),
        ((coluna + 1) * passo, (linha + 1) * passo),
        (70, 225, 255),
        -1,
    )
    for i in range(celulas + 1):
        cv2.line(tela, (i * passo, 0), (i * passo, tamanho), (130, 130, 130), 1)
        cv2.line(tela, (0, i * passo), (tamanho, i * passo), (130, 130, 130), 1)
    cv2.rectangle(tela, caixa[:2], caixa[2:], (20, 90, 220), 5)
    cv2.circle(tela, (cx, cy), 10, (0, 0, 190), -1)
    cv2.putText(tela, "centro define a celula responsavel", (90, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    return tela


def inferir_ultralytics(imagem_path: Path, output_dir: Path, modelo: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as erro:
        raise RuntimeError('Instale com: python -m pip install -e ".[yolo]"') from erro
    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada ou inválida: {imagem_path}")
    detector = YOLO(modelo)
    resultado = detector.predict(source=imagem, conf=0.50, verbose=False)[0]
    anotada = resultado.plot()
    salvar(output_dir / "02_inferencia_yolo.png", anotada)
    print(f"Objetos detectados: {len(resultado.boxes)}")


def executar(output_dir, imagem_path: Path | None = None, modelo: str = "yolov8n.pt") -> None:
    output_dir = preparar_saida(output_dir)
    salvar(output_dir / "01_responsabilidade_celular.png", diagrama_grid())
    caixa_a = (100, 100, 300, 300)
    caixa_b = (130, 120, 310, 315)
    print(f"IoU das caixas de exemplo: {iou_xyxy(caixa_a, caixa_b):.3f}")
    if imagem_path is not None:
        inferir_ultralytics(imagem_path, output_dir, modelo)
    else:
        print("[INFO] Informe --imagem para executar inferência real com Ultralytics.")


def main() -> None:
    parser = criar_parser("cap18", __doc__ or "")
    parser.add_argument("--imagem", type=Path, help="Imagem para inferência opcional.")
    parser.add_argument("--modelo", default="yolov8n.pt", help="Nome ou caminho do modelo YOLO.")
    args = parser.parse_args()
    executar(args.output_dir, args.imagem, args.modelo)


if __name__ == "__main__":
    main()
