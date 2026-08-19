"""Funções compartilhadas pelos exemplos.

Os exemplos priorizam reprodutibilidade: usam sementes fixas, criam entradas
sintéticas e salvam resultados em vez de depender de interface gráfica.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def criar_parser(capitulo: str, descricao: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / capitulo,
        help="Pasta em que as imagens produzidas serão gravadas.",
    )
    return parser


def preparar_saida(caminho: Path) -> Path:
    caminho.mkdir(parents=True, exist_ok=True)
    return caminho


def salvar(caminho: Path, imagem: np.ndarray) -> None:
    if imagem is None or imagem.size == 0:
        raise ValueError(f"Imagem vazia; não foi possível salvar {caminho}")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(caminho), imagem):
        raise OSError(f"O OpenCV não conseguiu gravar {caminho}")
    print(f"[OK] {caminho}")


def para_bgr(imagem: np.ndarray) -> np.ndarray:
    if imagem.ndim == 2:
        return cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    if imagem.ndim == 3 and imagem.shape[2] == 4:
        return cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
    return imagem.copy()


def mosaico(imagens: list[np.ndarray], colunas: int = 3, largura: int = 320) -> np.ndarray:
    """Normaliza tamanhos e organiza imagens em uma grade BGR."""
    if not imagens:
        raise ValueError("O mosaico precisa receber ao menos uma imagem.")

    normalizadas: list[np.ndarray] = []
    for imagem in imagens:
        bgr = para_bgr(imagem)
        escala = largura / bgr.shape[1]
        altura = max(1, round(bgr.shape[0] * escala))
        normalizadas.append(cv2.resize(bgr, (largura, altura), interpolation=cv2.INTER_AREA))

    altura_celula = max(imagem.shape[0] for imagem in normalizadas)
    linhas = (len(normalizadas) + colunas - 1) // colunas
    tela = np.full((linhas * altura_celula, colunas * largura, 3), 245, dtype=np.uint8)

    for indice, imagem in enumerate(normalizadas):
        linha, coluna = divmod(indice, colunas)
        y = linha * altura_celula
        x = coluna * largura
        tela[y : y + imagem.shape[0], x : x + imagem.shape[1]] = imagem
    return tela


def rotular(imagem: np.ndarray, texto: str) -> np.ndarray:
    saida = para_bgr(imagem)
    cv2.rectangle(saida, (0, 0), (saida.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        saida,
        texto,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return saida
