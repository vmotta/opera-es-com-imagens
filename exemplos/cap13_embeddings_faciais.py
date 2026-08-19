"""Capítulo 13: normalização e comparação de embeddings faciais sintéticos."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def normalizar_l2(vetor: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(vetor)
    if norma == 0:
        raise ValueError("Um embedding nulo não pode ser normalizado.")
    return vetor / norma


def distancia_euclidiana(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def similaridade_cosseno(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    gerador = np.random.default_rng(13)
    identidade_a = normalizar_l2(gerador.normal(size=128))
    amostra_a2 = normalizar_l2(identidade_a + gerador.normal(0, 0.025, size=128))
    identidade_b = normalizar_l2(gerador.normal(size=128))

    comparacoes = [
        ("A x A (mesma identidade sintética)", identidade_a, amostra_a2),
        ("A x B (identidades sintéticas distintas)", identidade_a, identidade_b),
    ]
    limiar_didatico = 0.60

    tela = np.full((330, 900, 3), 248, dtype=np.uint8)
    cv2.putText(tela, "COMPARACAO DE EMBEDDINGS 128D", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)
    for linha, (rotulo, vetor_1, vetor_2) in enumerate(comparacoes):
        distancia = distancia_euclidiana(vetor_1, vetor_2)
        cosseno = similaridade_cosseno(vetor_1, vetor_2)
        decisao = "MATCH" if distancia < limiar_didatico else "NAO MATCH"
        cor = (20, 150, 20) if decisao == "MATCH" else (30, 30, 200)
        y = 120 + linha * 110
        cv2.putText(tela, rotulo, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
        cv2.putText(
            tela,
            f"distancia={distancia:.3f}  cosseno={cosseno:.3f}  => {decisao}",
            (30, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            cor,
            2,
        )
        print(f"{rotulo}: distância={distancia:.3f}, cosseno={cosseno:.3f}, {decisao}")

    cv2.putText(
        tela,
        "Limiar 0.60 e apenas didatico; calibre com dados representativos.",
        (30, 305),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (90, 60, 20),
        2,
    )
    salvar(output_dir / "01_comparacao_embeddings.png", tela)


def main() -> None:
    parser = criar_parser("cap13", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
