"""Capítulo 8: pontos ORB, descritores binários, matching e homografia."""

from __future__ import annotations

import cv2
import numpy as np

from exemplos.comum import criar_parser, preparar_saida, salvar


def criar_referencia() -> np.ndarray:
    imagem = np.zeros((300, 300), dtype=np.uint8)
    cv2.rectangle(imagem, (25, 25), (275, 275), 230, 4)
    cv2.line(imagem, (40, 245), (250, 55), 180, 5)
    cv2.circle(imagem, (85, 85), 32, 255, -1)
    cv2.putText(imagem, "ORB", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4)
    for x in range(40, 280, 40):
        cv2.circle(imagem, (x, 260), 4, 255, -1)
    return imagem


def executar(output_dir) -> None:
    output_dir = preparar_saida(output_dir)
    referencia = criar_referencia()
    centro = (referencia.shape[1] / 2, referencia.shape[0] / 2)
    matriz = cv2.getRotationMatrix2D(centro, 24, 0.86)
    matriz[:, 2] += (95, 55)
    cena = cv2.warpAffine(referencia, matriz, (520, 430), borderValue=30)
    cv2.rectangle(cena, (400, 40), (480, 120), 170, -1)

    orb = cv2.ORB_create(nfeatures=900)
    kp1, des1 = orb.detectAndCompute(referencia, None)
    kp2, des2 = orb.detectAndCompute(cena, None)
    if des1 is None or des2 is None:
        raise RuntimeError("ORB não encontrou descritores; acrescente textura à imagem.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = matcher.knnMatch(des1, des2, k=2)
    # Teste de Lowe: aceita o melhor vizinho somente quando ele é claramente
    # mais próximo que o segundo. Isso reduz correspondências ambíguas.
    bons = [primeiro for primeiro, segundo in pares if primeiro.distance < 0.75 * segundo.distance]
    bons.sort(key=lambda correspondencia: correspondencia.distance)
    print(f"Keypoints: referência={len(kp1)}, cena={len(kp2)}, matches bons={len(bons)}")

    visual_keypoints = cv2.drawKeypoints(
        referencia,
        kp1,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
    visual_matches = cv2.drawMatches(
        referencia,
        kp1,
        cena,
        kp2,
        bons[:30],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cena_localizada = cv2.cvtColor(cena, cv2.COLOR_GRAY2BGR)
    if len(bons) >= 4:
        pontos_origem = np.float32([kp1[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
        pontos_destino = np.float32([kp2[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
        homografia, mascara_inliers = cv2.findHomography(
            pontos_origem,
            pontos_destino,
            cv2.RANSAC,
            5.0,
        )
        if homografia is not None:
            h, w = referencia.shape
            cantos = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            projetados = cv2.perspectiveTransform(cantos, homografia)
            cv2.polylines(cena_localizada, [np.int32(projetados)], True, (0, 255, 0), 4)
            print(f"Inliers RANSAC: {int(mascara_inliers.sum())}/{len(bons)}")

    salvar(output_dir / "01_keypoints_orb.png", visual_keypoints)
    salvar(output_dir / "02_correspondencias.png", visual_matches)
    salvar(output_dir / "03_objeto_localizado.png", cena_localizada)


def main() -> None:
    parser = criar_parser("cap08", __doc__ or "")
    executar(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
