"""Landmark-Extraktion aus Frame-Sequenzen – VisionFit AI v0.3.

Verarbeitet eine Liste von BGR-Frames und extrahiert für jeden Frame
die MediaPipe Face-Landmarks. Frames ohne erkanntes Gesicht erhalten
None als Landmark-Eintrag.

Das Modul ist vollständig Streamlit-unabhängig und pytest-testbar.
Eine gemeinsame FaceMesh-Instanz wird für alle Frames wiederverwendet
(Video-Modus: static_image_mode=False), was bei Sequenzen effizienter ist.

WICHTIGE HINWEISE (Limitationen):
    - Heuristische Analyse, kein Medizinprodukt, keine Diagnose.
    - Erkennungsqualität stark abhängig von Beleuchtung und Kamerawinkel.
    - Stabilitätsanalyse abhängig von Distanz zur Kamera.
    - Gesichtserkennung kann bei schnellen Bewegungen oder schlechtem Licht versagen.
    - Bei sehr niedrigen face_detection_rate-Werten (< 0.3) ist das Ergebnis
      als nicht zuverlässig einzustufen.
"""

from __future__ import annotations

import numpy as np

from app.cv.face_mesh import init_face_mesh, detect_face_landmarks


# ──────────────────────────────────────────────────────────────────────────────
# Öffentliche API
# ──────────────────────────────────────────────────────────────────────────────

def extract_landmarks_from_frames(
    frames: list[np.ndarray],
) -> dict:
    """Extrahiert Gesichts-Landmarks aus einer Frame-Sequenz.

    Erstellt eine gemeinsame FaceMesh-Instanz im Video-Modus
    (static_image_mode=False) für alle Frames. Dies ist effizienter
    als pro Frame eine neue Instanz zu erzeugen.

    Args:
        frames: Liste von BGR-Frames als numpy Arrays (H, W, 3).
                Ungültige Frames (None, falsche Shape) werden als
                "kein Gesicht erkannt" behandelt.

    Returns:
        Dictionary mit:
            - frames_landmarks (list): Pro Frame eine Landmark-Liste
              (x, y, z)-Tupel in normierten Koordinaten [0.0, 1.0],
              oder None wenn kein Gesicht erkannt wurde.
            - face_detection_rate (float): Anteil der Frames mit
              erkanntem Gesicht (0.0–1.0, 3 Dezimalstellen).

    Hinweis:
        face_detection_rate < 0.3 deutet auf unzuverlässige Daten hin
        (schlechte Beleuchtung, Gesicht nicht vollständig sichtbar etc.).

    Beispiel:
        >>> black = [np.zeros((480, 640, 3), dtype=np.uint8)] * 5
        >>> result = extract_landmarks_from_frames(black)
        >>> result["face_detection_rate"]
        0.0
        >>> len(result["frames_landmarks"])
        5
    """
    if not frames:
        return {"frames_landmarks": [], "face_detection_rate": 0.0}

    # Video-Modus: static_image_mode=False ist effizienter für Sequenzen,
    # da MediaPipe Tracking zwischen Frames nutzt statt jeden Frame neu zu detektieren.
    face_mesh = init_face_mesh(static_image_mode=False)

    frames_landmarks: list = []
    detected_count: int = 0

    for frame in frames:
        # Ungültige Frames defensiv behandeln
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            frames_landmarks.append(None)
            continue

        result = detect_face_landmarks(frame, face_mesh=face_mesh)

        if result["face_detected"] and result["landmarks"]:
            frames_landmarks.append(result["landmarks"])
            detected_count += 1
        else:
            frames_landmarks.append(None)

    total = len(frames)
    rate = round(detected_count / total, 3) if total > 0 else 0.0

    return {
        "frames_landmarks": frames_landmarks,
        "face_detection_rate": rate,
    }
