"""Capítulo 18: grade didática, letterbox, coordenadas, métricas e YOLO opcional."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersecao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uniao = area_a + area_b - intersecao
    return float(intersecao / uniao) if uniao else 0.0


def letterbox(imagem: np.ndarray, alvo=(640, 640), cor=(114, 114, 114)):
    """Preserva proporção, adiciona padding e devolve parâmetros para inversão."""
    h, w = imagem.shape[:2]
    alvo_w, alvo_h = alvo
    escala = min(alvo_w / w, alvo_h / h)
    novo_w = int(round(w * escala))
    novo_h = int(round(h * escala))
    red = cv2.resize(imagem, (novo_w, novo_h), interpolation=cv2.INTER_LINEAR)
    esquerda = (alvo_w - novo_w) // 2
    direita = alvo_w - novo_w - esquerda
    topo = (alvo_h - novo_h) // 2
    baixo = alvo_h - novo_h - topo
    saida = cv2.copyMakeBorder(
        red,
        topo,
        baixo,
        esquerda,
        direita,
        cv2.BORDER_CONSTANT,
        value=cor,
    )
    return saida, float(escala), esquerda, topo


def caixa_para_letterbox(caixa, escala, pad_x, pad_y):
    x1, y1, x2, y2 = caixa
    return (
        x1 * escala + pad_x,
        y1 * escala + pad_y,
        x2 * escala + pad_x,
        y2 * escala + pad_y,
    )


def desfazer_letterbox(caixa, escala, pad_x, pad_y, largura, altura):
    x1, y1, x2, y2 = caixa
    resultado = [
        (x1 - pad_x) / escala,
        (y1 - pad_y) / escala,
        (x2 - pad_x) / escala,
        (y2 - pad_y) / escala,
    ]
    resultado[0] = np.clip(resultado[0], 0, largura - 1)
    resultado[1] = np.clip(resultado[1], 0, altura - 1)
    resultado[2] = np.clip(resultado[2], 0, largura - 1)
    resultado[3] = np.clip(resultado[3], 0, altura - 1)
    return tuple(float(v) for v in resultado)


def diagrama_grid() -> np.ndarray:
    tamanho, celulas = 700, 7
    tela = np.full((tamanho, tamanho, 3), 245, dtype=np.uint8)
    passo = tamanho // celulas
    caixa = (240, 190, 555, 505)
    cx, cy = (caixa[0] + caixa[2]) // 2, (caixa[1] + caixa[3]) // 2
    coluna, linha = cx // passo, cy // passo
    cv2.rectangle(tela, (coluna * passo, linha * passo), ((coluna + 1) * passo, (linha + 1) * passo), (70, 225, 255), -1)
    for i in range(celulas + 1):
        cv2.line(tela, (i * passo, 0), (i * passo, tamanho), (130, 130, 130), 1)
        cv2.line(tela, (0, i * passo), (tamanho, i * passo), (130, 130, 130), 1)
    cv2.rectangle(tela, caixa[:2], caixa[2:], (20, 90, 220), 5)
    cv2.circle(tela, (cx, cy), 10, (0, 0, 190), -1)
    cv2.putText(tela, f"centro=({cx},{cy}) -> celula (linha={linha}, coluna={coluna})", (55, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    return tela


def inferir_ultralytics(imagem_path: Path, output_dir: Path, modelo: str, conf: float) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as erro:
        raise RuntimeError('Instale com: python -m pip install -e ".[yolo]"') from erro

    imagem = cv2.imread(str(imagem_path))
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada ou inválida: {imagem_path}")

    detector = YOLO(modelo)
    # Aquecimento simples; evita usar a primeira chamada como benchmark isolado.
    detector.predict(source=imagem, conf=conf, verbose=False)
    tempos = []
    resultado = None
    for _ in range(6):
        inicio = time.perf_counter()
        resultado = detector.predict(source=imagem, conf=conf, verbose=False)[0]
        tempos.append(time.perf_counter() - inicio)
    assert resultado is not None

    anotada = resultado.plot()
    salvar(output_dir / "02_inferencia_yolo.png", anotada)
    mediana_ms = float(np.median(tempos) * 1000)
    print(f"Objetos detectados: {len(resultado.boxes)}")
    print(f"Latência mediana de predict (6 execuções após aquecimento): {mediana_ms:.2f} ms")

    # Inspeção defensiva das caixas se existirem.
    if len(resultado.boxes):
        xyxy = resultado.boxes.xyxy.detach().cpu().numpy()
        confs = resultado.boxes.conf.detach().cpu().numpy()
        cls = resultado.boxes.cls.detach().cpu().numpy().astype(int)
        for i in range(min(5, len(xyxy))):
            print(f"det[{i}] classe={cls[i]} conf={confs[i]:.3f} xyxy={xyxy[i].round(1).tolist()}")


def executar(output_dir, imagem_path: Path | None = None, modelo: str = "yolov8n.pt", conf: float = 0.50) -> None:
    output_dir = preparar_saida(output_dir)

    # --------------------------------------------------------------------------
    # 1. RESPONSABILIDADE ESPACIAL CLÁSSICA — APENAS MODELO DIDÁTICO.
    # --------------------------------------------------------------------------
    grade = diagrama_grid()
    salvar(output_dir / "01_responsabilidade_celular.png", grade)

    # --------------------------------------------------------------------------
    # 2. LETTERBOX E INVERSÃO EXATA DE COORDENADAS.
    # --------------------------------------------------------------------------
    original = np.full((720, 1280, 3), 225, dtype=np.uint8)
    caixa_original = (310.0, 160.0, 955.0, 610.0)
    cv2.rectangle(original, (310, 160), (955, 610), (30, 110, 220), 6)
    cv2.putText(original, "objeto no espaço original 1280x720", (320, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

    preparada, escala, pad_x, pad_y = letterbox(original, (640, 640))
    caixa_preparada = caixa_para_letterbox(caixa_original, escala, pad_x, pad_y)
    cp = tuple(int(round(v)) for v in caixa_preparada)
    cv2.rectangle(preparada, cp[:2], cp[2:], (0, 255, 0), 4)

    recuperada = desfazer_letterbox(
        caixa_preparada,
        escala,
        pad_x,
        pad_y,
        original.shape[1],
        original.shape[0],
    )
    erro = np.abs(np.array(recuperada) - np.array(caixa_original))
    print(f"Letterbox: escala={escala:.4f}, pad_x={pad_x}, pad_y={pad_y}")
    print("Caixa original:", caixa_original)
    print("Caixa preparada:", tuple(round(v, 2) for v in caixa_preparada))
    print("Caixa recuperada:", tuple(round(v, 4) for v in recuperada))
    print(f"Erro máximo ida-volta: {erro.max():.6f} px")

    # Comparação com resize direto que distorce a razão de aspecto.
    deformada = cv2.resize(original, (640, 640), interpolation=cv2.INTER_LINEAR)

    # --------------------------------------------------------------------------
    # 3. IoU + PRECISÃO/REVOCAÇÃO EM EXEMPLO NUMÉRICO.
    # --------------------------------------------------------------------------
    caixa_a = (100, 100, 300, 300)
    caixa_b = (130, 120, 310, 315)
    print(f"IoU das caixas de exemplo: {iou_xyxy(caixa_a, caixa_b):.3f}")
    tp, fp, fn = 80, 20, 40
    precisao = tp / (tp + fp)
    revocacao = tp / (tp + fn)
    print(f"Exemplo de métricas: TP={tp}, FP={fp}, FN={fn} -> precisão={precisao:.3f}, revocação={revocacao:.3f}")

    # --------------------------------------------------------------------------
    # 4. BENCHMARK DA ETAPA DE LETTERBOX, SEM DEPENDER DE GPU/REDE.
    # --------------------------------------------------------------------------
    tempos = []
    for _ in range(30):
        inicio = time.perf_counter()
        letterbox(original, (640, 640))
        tempos.append(time.perf_counter() - inicio)
    print(f"Letterbox mediana (30 execuções): {np.median(tempos) * 1000:.3f} ms")

    salvar(output_dir / "02_original_com_caixa.png", original)
    salvar(output_dir / "03_letterbox_640.png", preparada)
    salvar(output_dir / "04_resize_direto_deformado.png", deformada)

    painel = mosaico(
        [
            rotular(grade, "Grade YOLO clássica"),
            rotular(original, "Original 1280x720"),
            rotular(preparada, "Letterbox 640x640"),
            rotular(deformada, "Resize direto 640x640"),
        ],
        colunas=2,
    )
    salvar(output_dir / "painel.png", painel)

    if imagem_path is not None:
        inferir_ultralytics(imagem_path, output_dir, modelo, conf)
    else:
        print("[INFO] Informe --imagem para executar inferência real com Ultralytics; o laboratório geométrico já foi concluído.")


def main() -> None:
    parser = criar_parser("cap18", __doc__ or "")
    parser.add_argument("--imagem", type=Path, help="Imagem para inferência opcional.")
    parser.add_argument("--modelo", default="yolov8n.pt", help="Nome ou caminho do modelo YOLO.")
    parser.add_argument("--conf", type=float, default=0.50, help="Limiar de confiança para inferência opcional.")
    args = parser.parse_args()
    executar(args.output_dir, args.imagem, args.modelo, args.conf)


if __name__ == "__main__":
    main()
