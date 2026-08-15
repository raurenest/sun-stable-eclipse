"""
centrar_sol.py
--------------
Pase 1: detección directa del centro del sol (versión estable).
Pase 2: correlación de fase incremental para eliminar tembleque residual.

Uso:
    python centrar_sol.py input.mp4 output.mp4

Dependencias:
    pip install opencv-python numpy tqdm
"""

import cv2
import numpy as np
import sys
import tempfile
import os
from tqdm import tqdm

ESCALA_DETECCION = 1.0
UMBRAL_CONFIANZA = 0.7
UMBRAL_RADIO_PARCIAL = 440  # ~110 * 4 para escala 100%
MAX_DESPLAZAMIENTO_CORR = 10
ESCALA_CORRELACION = 0.5  # más resolución para correlación más precisa


def detectar_sol(pequeño):
    h, w = pequeño.shape[:2]
    gris = cv2.cvtColor(pequeño, cv2.COLOR_BGR2GRAY)

    # Blur gaussiano para eliminar artefactos de compresión antes de umbralizar
    gris = cv2.GaussianBlur(gris, (9, 9), 0)
    _, thresh = cv2.threshold(gris, 40, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        c = max(contornos, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_c = int(M["m10"] / M["m00"])
            cy_c = int(M["m01"] / M["m00"])
            hull = cv2.convexHull(c)
            puntos = hull.reshape(-1, 2).astype(np.float32)
            (cx_e, cy_e), radio = cv2.minEnclosingCircle(puntos)
            cx_e, cy_e, radio = int(cx_e), int(cy_e), int(radio)
            if 10 <= radio <= min(h, w) * 0.75:
                dist = np.sqrt((cx_e - cx_c)**2 + (cy_e - cy_c)**2)
                if dist / radio < UMBRAL_CONFIANZA and radio <= UMBRAL_RADIO_PARCIAL:
                    return cx_e, cy_e

    _, corona = cv2.threshold(gris, 15, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((9, 9), np.uint8)
    corona = cv2.morphologyEx(corona, cv2.MORPH_CLOSE, kernel2)
    borde = np.zeros((h+2, w+2), np.uint8)
    flood = corona.copy()
    cv2.floodFill(flood, borde, (0, 0), 255)
    hueco = cv2.bitwise_not(flood)
    contornos2, _ = cv2.findContours(hueco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos2:
        luna = max(contornos2, key=cv2.contourArea)
        if cv2.contourArea(luna) >= (min(h, w) * 0.1) ** 2:
            hull2 = cv2.convexHull(luna)
            puntos2 = hull2.reshape(-1, 2).astype(np.float32)
            (cx, cy), radio = cv2.minEnclosingCircle(puntos2)
            if min(h, w) * 0.1 <= radio <= min(h, w) * 0.7:
                return int(cx), int(cy)

    return None


def pase1_deteccion(ruta_entrada, ruta_intermedia):
    cap = cv2.VideoCapture(ruta_entrada)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_det = int(w * ESCALA_DETECCION)
    h_det = int(h * ESCALA_DETECCION)

    print(f"Vídeo: {w}x{h} @ {fps:.1f} fps, {total_frames} frames")
    print("\n── Pase 1: detección directa ──")
    print("  Detectando centros...")

    centros = []
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        pequeño = cv2.resize(frame, (w_det, h_det), interpolation=cv2.INTER_AREA)
        det = detectar_sol(pequeño)
        if det is not None:
            centros.append((int(det[0] / ESCALA_DETECCION),
                            int(det[1] / ESCALA_DETECCION)))
        else:
            centros.append(None)
    cap.release()

    detectados = [c for c in centros if c is not None]
    cx_medio = int(np.mean([c[0] for c in detectados]))
    cy_medio = int(np.mean([c[1] for c in detectados]))
    print(f"  Detectado en {len(detectados)}/{len(centros)} frames")

    # Rellenar huecos con interpolación
    n = len(centros)
    centros_finales = list(centros)
    for i in range(n):
        if centros_finales[i] is None:
            ant = next((centros_finales[j] for j in range(i-1, -1, -1) if centros_finales[j] is not None), None)
            sig = next((centros_finales[j] for j in range(i+1, n) if centros_finales[j] is not None), None)
            if ant and sig:
                centros_finales[i] = ((ant[0]+sig[0])//2, (ant[1]+sig[1])//2)
            elif ant:
                centros_finales[i] = ant
            elif sig:
                centros_finales[i] = sig
            else:
                centros_finales[i] = (cx_medio, cy_medio)

    print("  Generando vídeo intermedio...")
    cap2 = cv2.VideoCapture(ruta_entrada)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_intermedia, fourcc, fps, (w, h))
    cx_obj, cy_obj = w // 2, h // 2

    for i in tqdm(range(len(centros_finales))):
        ret, frame = cap2.read()
        if not ret:
            break
        cx, cy = centros_finales[i]
        M = np.float32([[1, 0, cx_obj - cx], [0, 1, cy_obj - cy]])
        out.write(cv2.warpAffine(frame, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0)))
    cap2.release()
    out.release()
    print(f"  ✓ Vídeo intermedio guardado")
    return w, h, fps


def pase2_correlacion(ruta_intermedia, ruta_salida, w, h, fps):
    cap = cv2.VideoCapture(ruta_intermedia)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_corr = int(w * ESCALA_CORRELACION)
    h_corr = int(h * ESCALA_CORRELACION)

    print("\n── Pase 2: correlación de fase ──")
    print(f"  Resolución de correlación: {w_corr}x{h_corr}")
    ventana = cv2.createHanningWindow((w_corr, h_corr), cv2.CV_32F)
    max_det = MAX_DESPLAZAMIENTO_CORR * ESCALA_CORRELACION

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (w, h))

    ret, frame_prev = cap.read()
    out.write(frame_prev)

    prev_det = cv2.resize(frame_prev, (w_corr, h_corr), interpolation=cv2.INTER_AREA)
    prev_gris = cv2.cvtColor(prev_det, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_ventana = prev_gris * ventana
    dx_acum, dy_acum = 0.0, 0.0

    print("  Estabilizando...")
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break

        curr_corr = cv2.resize(frame, (w_corr, h_corr), interpolation=cv2.INTER_AREA)
        curr_gris = cv2.cvtColor(curr_corr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_ventana = curr_gris * ventana

        (dx_det, dy_det), _ = cv2.phaseCorrelate(prev_ventana, curr_ventana)

        if abs(dx_det) <= max_det and abs(dy_det) <= max_det:
            dx_acum += dx_det / ESCALA_CORRELACION
            dy_acum += dy_det / ESCALA_CORRELACION

        M = np.float32([[1, 0, -dx_acum], [0, 1, -dy_acum]])
        frame_corr = cv2.warpAffine(frame, M, (w, h),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
        out.write(frame_corr)

        prev_corr2 = cv2.resize(frame_corr, (w_corr, h_corr), interpolation=cv2.INTER_AREA)
        prev_gris = cv2.cvtColor(prev_corr2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev_ventana = prev_gris * ventana

    cap.release()
    out.release()
    print(f"  ✓ Vídeo final: {ruta_salida}")


def procesar_video(ruta_entrada, ruta_salida):
    tmp = tempfile.mktemp(suffix='.mp4')
    try:
        w, h, fps = pase1_deteccion(ruta_entrada, tmp)
        pase2_correlacion(tmp, ruta_salida, w, h, fps)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print("\n✓ Proceso completo.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python centrar_sol.py input.mp4 output.mp4")
        sys.exit(1)
    procesar_video(sys.argv[1], sys.argv[2])
