from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

MODULOS = [
    "cap01_fundamentos",
    "cap02_transformacoes",
    "cap03_filtros_bordas",
    "cap04_limiar_morfologia",
    "cap05_contornos",
    "cap06_espacos_cores",
    "cap07_template_haar",
    "cap08_feature_matching",
    "cap09_fluxo_optico",
    "cap10_dnn_pipeline",
    "cap11_yolo_nms",
    "cap12_pose",
    "cap13_embeddings_faciais",
    "cap14_ocr",
    "cap15_visao_estereo",
    "cap16_cnn",
    "cap17_gan",
    "cap18_yolo_moderno",
]


@pytest.mark.parametrize("nome", MODULOS)
def test_exemplo_gera_ao_menos_uma_imagem(nome: str, tmp_path: Path) -> None:
    modulo = importlib.import_module(f"exemplos.{nome}")
    modulo.executar(tmp_path / nome)
    imagens = list((tmp_path / nome).glob("*.png"))
    assert imagens, f"{nome} não gerou imagem de saída"
    assert all(arquivo.stat().st_size > 100 for arquivo in imagens)


def test_normalizacao_embedding() -> None:
    from exemplos.cap13_embeddings_faciais import normalizar_l2

    resultado = normalizar_l2(np.array([3.0, 4.0]))
    assert np.linalg.norm(resultado) == pytest.approx(1.0)


def test_iou() -> None:
    from exemplos.cap18_yolo_moderno import iou_xyxy

    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
