"""Capítulo 8: ORB, descritores binários, matching, teste de razão e RANSAC."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, mosaico, preparar_saida, rotular, salvar


def criar_referencia() -> np.ndarray:
    imagem = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(imagem, (25, 25), (275, 275), 230, 4)
    cv2.line(imagem, (40, 245), (250, 55), 180, 5)
    cv2.circle(imagem, (85, 85), 32, 255, -1)
    cv2.putText(imagem, "ORB", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4)
    for x in range(40, 280, 40):
        cv2.circle(imagem, (x, 260), 4, 255, -1)
    for y in range(45, 245, 35):
        cv2.rectangle(imagem, (235, y), (246, y + 11), 150 + (y % 70), -1)
    return imagem


def filtrar_razao(pares, razao: float):
    """Aplica o teste de razão aos dois vizinhos mais próximos."""
    bons = []
    for par in pares:
        if len(par) < 2:
            continue
        primeiro, segundo = par
        if primeiro.distance < razao * segundo.distance:
            bons.append(primeiro)
    return sorted(bons, key=lambda m: m.distance)


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    referencia = criar_referencia()

    # Cena com rotação, escala e translação.
    centro = (referencia.shape[1] / 2, referencia.shape[0] / 2)
    matriz = cv2.getRotationMatrix2D(centro, 24, 0.86)
    matriz[:, 2] += (95, 55)
    cena = cv2.warpAffine(referencia, matriz, (520, 430), borderValue=30)

    # Padrão repetitivo proposital para introduzir matches ambíguos.
    for x in range(395, 490, 24):
        for y in range(35, 130, 24):
            cv2.rectangle(cena, (x, y), (x + 10, y + 10), 190, -1)

    # --------------------------------------------------------------------------
    # 1. DETECÇÃO E DESCRIÇÃO ORB.
    # --------------------------------------------------------------------------
    orb = cv2.ORB_create(nfeatures=1200)
    kp1, des1 = orb.detectAndCompute(referencia, None)
    kp2, des2 = orb.detectAndCompute(cena, None)
    if des1 is None or des2 is None:
        raise RuntimeError("ORB não encontrou descritores; acrescente textura à imagem.")

    print(f"Keypoints: referência={len(kp1)}, cena={len(kp2)}")
    print(f"Descritores: ref={des1.shape} dtype={des1.dtype}; cena={des2.shape}")

    visual_keypoints_ref = cv2.drawKeypoints(
        referencia,
        kp1,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
    visual_keypoints_cena = cv2.drawKeypoints(
        cena,
        kp2,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )

    # --------------------------------------------------------------------------
    # 2. KNN + TESTE DE RAZÃO EM TRÊS NÍVEIS.
    # --------------------------------------------------------------------------
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = matcher.knnMatch(des1, des2, k=2)
    conjuntos = {}
    for razao in (0.60, 0.75, 0.90):
        bons = filtrar_razao(pares, razao)
        conjuntos[razao] = bons
        print(f"Teste de razão {razao:.2f}: {len(bons)} matches")

    bons = conjuntos[0.75]

    # --------------------------------------------------------------------------
    # 3. CROSS-CHECK PARA COMPARAÇÃO DIDÁTICA.
    # --------------------------------------------------------------------------
    matcher_cross = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_cross = sorted(matcher_cross.match(des1, des2), key=lambda m: m.distance)
    print(f"Cross-check: {len(matches_cross)} matches recíprocos")

    visual_ratio_60 = cv2.drawMatches(
        referencia, kp1, cena, kp2, conjuntos[0.60][:35], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    visual_ratio_75 = cv2.drawMatches(
        referencia, kp1, cena, kp2, bons[:35], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    visual_ratio_90 = cv2.drawMatches(
        referencia, kp1, cena, kp2, conjuntos[0.90][:35], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    visual_cross = cv2.drawMatches(
        referencia, kp1, cena, kp2, matches_cross[:35], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # --------------------------------------------------------------------------
    # 4. HOMOGRAFIA E RANSAC.
    # --------------------------------------------------------------------------
    cena_localizada = cv2.cvtColor(cena, cv2.COLOR_GRAY2BGR)
    visual_inliers = visual_ratio_75.copy()
    if len(bons) >= 4:
        pontos_origem = np.float32([kp1[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
        pontos_destino = np.float32([kp2[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
        homografia, mascara_inliers = cv2.findHomography(
            pontos_origem,
            pontos_destino,
            cv2.RANSAC,
            5.0,
        )

        if homografia is not None and mascara_inliers is not None:
            mascara_bool = mascara_inliers.ravel().astype(bool)
            inliers = [m for m, ok in zip(bons, mascara_bool) if ok]
            quantidade_inliers = int(mascara_bool.sum())
            taxa = quantidade_inliers / len(bons)
            print(f"Inliers RANSAC: {quantidade_inliers}/{len(bons)} ({taxa:.1%})")
            print("Homografia:\n", homografia)

            visual_inliers = cv2.drawMatches(
                referencia,
                kp1,
                cena,
                kp2,
                inliers[:40],
                None,
                matchColor=(0, 255, 0),
                singlePointColor=(0, 0, 255),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )

            h, w = referencia.shape
            cantos = np.float32(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
            ).reshape(-1, 1, 2)
            projetados = cv2.perspectiveTransform(cantos, homografia)
            cv2.polylines(cena_localizada, [np.int32(projetados)], True, (0, 255, 0), 4)
            cv2.putText(
                cena_localizada,
                f"inliers {quantidade_inliers}/{len(bons)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            print("RANSAC não conseguiu estimar uma homografia válida.")
    else:
        print("Menos de quatro bons matches: homografia não pode ser estimada.")

    salvar(output_dir / "01_keypoints_orb.png", visual_keypoints_ref)
    salvar(output_dir / "02_keypoints_cena.png", visual_keypoints_cena)
    salvar(output_dir / "03_razao_060.png", visual_ratio_60)
    salvar(output_dir / "04_razao_075.png", visual_ratio_75)
    salvar(output_dir / "05_razao_090.png", visual_ratio_90)
    salvar(output_dir / "06_cross_check.png", visual_cross)
    salvar(output_dir / "07_inliers_ransac.png", visual_inliers)
    salvar(output_dir / "08_objeto_localizado.png", cena_localizada)

    # Nomes históricos referenciados na documentação.
    salvar(output_dir / "02_correspondencias.png", visual_ratio_75)
    salvar(output_dir / "03_objeto_localizado.png", cena_localizada)

    painel = mosaico(
        [
            rotular(visual_keypoints_ref, "Keypoints referência"),
            rotular(visual_keypoints_cena, "Keypoints cena"),
            rotular(visual_ratio_60, "Razão 0.60"),
            rotular(visual_ratio_75, "Razão 0.75"),
            rotular(visual_ratio_90, "Razão 0.90"),
            rotular(visual_cross, "Cross-check"),
            rotular(visual_inliers, "Inliers RANSAC"),
            rotular(cena_localizada, "Homografia projetada"),
        ],
        colunas=2,
    )
    salvar(output_dir / "painel.png", painel)


def main() -> None:
    parser = criar_parser("cap08", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
