from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
IMAGEM = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@pytest.mark.parametrize("pagina", sorted(DOCS.rglob("*.md")), ids=lambda p: p.name)
def test_imagens_locais_existem_e_tem_texto_alternativo(pagina: Path) -> None:
    for texto_alternativo, destino in IMAGEM.findall(pagina.read_text(encoding="utf-8")):
        assert texto_alternativo.strip(), f"Imagem sem texto alternativo em {pagina}"
        if "://" not in destino:
            caminho = (pagina.parent / destino.split("#", 1)[0]).resolve()
            assert caminho.is_file(), f"Imagem local ausente em {pagina}: {destino}"
