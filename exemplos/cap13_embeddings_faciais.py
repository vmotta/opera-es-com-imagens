"""Capítulo 13: embeddings sintéticos, 1:1, 1:N, FAR, FRR e calibração didática."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def normalizar_l2(vetor: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(vetor)
    if norma == 0:
        raise ValueError("Um embedding nulo não pode ser normalizado.")
    return vetor / norma


def distancia_euclidiana(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def similaridade_cosseno(a: np.ndarray, b: np.ndarray) -> float:
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denominador == 0 else float(np.dot(a, b) / denominador)


def desenhar_distribuicoes(genuinos, impostores, limiar, tamanho=(900, 430)):
    largura, altura = tamanho
    tela = np.full((altura, largura, 3), 248, dtype=np.uint8)
    minimo = float(min(genuinos.min(), impostores.min()))
    maximo = float(max(genuinos.max(), impostores.max()))
    bins = np.linspace(minimo, maximo, 45)
    hg, _ = np.histogram(genuinos, bins=bins)
    hi, _ = np.histogram(impostores, bins=bins)
    max_hist = max(hg.max(), hi.max(), 1)
    x0, y_base = 55, altura - 55
    largura_plot = largura - 100
    altura_plot = altura - 105

    for hist, cor in [(hg, (40, 160, 40)), (hi, (40, 60, 210))]:
        pontos = []
        for i, valor in enumerate(hist):
            x = int(x0 + i / max(1, len(hist) - 1) * largura_plot)
            y = int(y_base - valor / max_hist * altura_plot)
            pontos.append((x, y))
        if len(pontos) > 1:
            cv2.polylines(tela, [np.array(pontos, dtype=np.int32)], False, cor, 3)

    x_limiar = int(x0 + (limiar - minimo) / max(1e-9, maximo - minimo) * largura_plot)
    cv2.line(tela, (x_limiar, 35), (x_limiar, y_base), (30, 30, 30), 2)
    cv2.putText(tela, f"limiar={limiar:.2f}", (max(5, x_limiar - 60), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    cv2.putText(tela, "verde: pares genuinos", (55, altura - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 130, 40), 2)
    cv2.putText(tela, "vermelho: impostores", (330, altura - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 60, 190), 2)
    return tela


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    rng = np.random.default_rng(13)

    # --------------------------------------------------------------------------
    # 1. COMPARAÇÃO 1:1.
    # --------------------------------------------------------------------------
    identidade_a = normalizar_l2(rng.normal(size=128))
    amostra_a2 = normalizar_l2(identidade_a + rng.normal(0, 0.025, size=128))
    identidade_b = normalizar_l2(rng.normal(size=128))

    comparacoes = [
        ("A x A (genuino sintetico)", identidade_a, amostra_a2),
        ("A x B (impostor sintetico)", identidade_a, identidade_b),
    ]
    limiar_didatico = 0.60
    tela = np.full((330, 900, 3), 248, dtype=np.uint8)
    cv2.putText(tela, "COMPARACAO DE EMBEDDINGS 128D", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2)
    for linha, (rotulo, vetor_1, vetor_2) in enumerate(comparacoes):
        distancia = distancia_euclidiana(vetor_1, vetor_2)
        cosseno = similaridade_cosseno(vetor_1, vetor_2)
        decisao = "MATCH" if distancia <= limiar_didatico else "NAO MATCH"
        cor = (20, 150, 20) if decisao == "MATCH" else (30, 30, 200)
        y = 120 + linha * 110
        cv2.putText(tela, rotulo, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
        cv2.putText(tela, f"dist={distancia:.3f} cos={cosseno:.3f} => {decisao}", (30, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, cor, 2)
        print(f"{rotulo}: distância={distancia:.3f}, cosseno={cosseno:.3f}, {decisao}")

    # --------------------------------------------------------------------------
    # 2. IDENTIFICAÇÃO 1:N E A CLASSE DESCONHECIDO.
    # --------------------------------------------------------------------------
    base = np.vstack([normalizar_l2(rng.normal(size=128)) for _ in range(80)])
    consulta_conhecida = normalizar_l2(base[17] + rng.normal(0, 0.025, size=128))
    consulta_desconhecida = normalizar_l2(rng.normal(size=128))

    def identificar(consulta):
        distancias = np.linalg.norm(base - consulta, axis=1)
        indice = int(np.argmin(distancias))
        melhor = float(distancias[indice])
        return indice if melhor <= limiar_didatico else None, melhor

    idx_k, d_k = identificar(consulta_conhecida)
    idx_u, d_u = identificar(consulta_desconhecida)
    print(f"1:N consulta conhecida: candidato={idx_k}, distância={d_k:.3f}")
    print(f"1:N consulta desconhecida: candidato={idx_u}, distância={d_u:.3f} (None significa rejeitada)")

    # --------------------------------------------------------------------------
    # 3. DISTRIBUIÇÕES GENUÍNAS E IMPOSTORAS SINTÉTICAS + FAR/FRR.
    # --------------------------------------------------------------------------
    n = 1200
    centros = [normalizar_l2(rng.normal(size=128)) for _ in range(n)]
    genuinos = np.empty(n, dtype=np.float64)
    impostores = np.empty(n, dtype=np.float64)
    for i, centro in enumerate(centros):
        a = normalizar_l2(centro + rng.normal(0, 0.035, size=128))
        b = normalizar_l2(centro + rng.normal(0, 0.035, size=128))
        genuinos[i] = distancia_euclidiana(a, b)
        outro = normalizar_l2(rng.normal(size=128))
        impostores[i] = distancia_euclidiana(a, outro)

    print("\nLimiar | FAR | FRR")
    tabela = []
    for limiar in np.linspace(0.35, 1.35, 11):
        far = float(np.mean(impostores <= limiar))
        frr = float(np.mean(genuinos > limiar))
        tabela.append((limiar, far, frr))
        print(f"{limiar:6.2f} | {far:5.3f} | {frr:5.3f}")

    # Escolhe limiar didático pela menor soma FAR+FRR — apenas neste experimento sintético.
    limiar_calibrado, far_cal, frr_cal = min(tabela, key=lambda linha: linha[1] + linha[2])
    print(f"Limiar sintético de menor FAR+FRR: {limiar_calibrado:.2f}; FAR={far_cal:.3f}; FRR={frr_cal:.3f}")
    distribuicoes = desenhar_distribuicoes(genuinos, impostores, limiar_calibrado)

    # Gráfico FAR/FRR por limiar.
    grafico = np.full((430, 900, 3), 248, dtype=np.uint8)
    pontos_far, pontos_frr = [], []
    for i, (_, far, frr) in enumerate(tabela):
        x = 55 + int(i / (len(tabela) - 1) * 790)
        pontos_far.append((x, 360 - int(far * 300)))
        pontos_frr.append((x, 360 - int(frr * 300)))
    cv2.polylines(grafico, [np.array(pontos_far, dtype=np.int32)], False, (40, 60, 210), 3)
    cv2.polylines(grafico, [np.array(pontos_frr, dtype=np.int32)], False, (40, 160, 40), 3)
    cv2.putText(grafico, "FAR (vermelho) e FRR (verde) ao variar limiar", (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    salvar(output_dir / "01_comparacao_embeddings.png", tela)
    salvar(output_dir / "02_distribuicoes_genuino_impostor.png", distribuicoes)
    salvar(output_dir / "03_far_frr.png", grafico)

    painel = mosaico(
        [
            rotular(tela, "Verificação 1:1"),
            rotular(distribuicoes, "Distribuições sintéticas"),
            rotular(grafico, "FAR / FRR"),
        ],
        colunas=3,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap13", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
