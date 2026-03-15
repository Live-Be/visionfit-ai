"""Video-Frame-Sequenz-Erfassung und Validierung – VisionFit AI v0.3.

Dieses Modul stellt reine Hilfsfunktionen für Frame-Verwaltung bereit.
Es ist vollständig Streamlit-unabhängig und pytest-testbar.

Für die browserbasierte Live-Kameraerfassung in der Streamlit-App wird
streamlit-webrtc in app/tests/fixation_test.py verwendet. Dieses Modul
dient dort zur Nachverarbeitung der gesammelten Frames.

capture_frame_sequence() nutzt OpenCV direkt (lokale Webcam, kein Browser).
Es eignet sich für Tests, CLI-Tools und Nicht-Browser-Umgebungen.

WICHTIGE HINWEISE (Limitationen):
    - Heuristische Analyse, kein Medizinprodukt, keine Diagnose.
    - Blinkrate und Bildqualität stark abhängig von Kamera und Beleuchtung.
    - Stabilitätsanalyse abhängig von Distanz zur Kamera und Aufnahmewinkel.
    - Für klinisch belastbare Auswertungen sind kalibrierte Geräte erforderlich.
    - Die FPS-Schätzung aus cv2.CAP_PROP_FPS kann geräteabhängig ungenau sein.
"""

from __future__ import annotations

import time

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

MAX_FRAMES: int = 150           # Maximale Frame-Anzahl pro Sequenz
DEFAULT_FPS: float = 30.0       # Standard-FPS wenn Kamera keinen Wert liefert
DEFAULT_CAPTURE_SECONDS: float = 3.0  # Standard-Aufnahmedauer


# ──────────────────────────────────────────────────────────────────────────────
# Kern-Funktionen (rein, ohne Seiteneffekte)
# ──────────────────────────────────────────────────────────────────────────────

def validate_frames(frames: list) -> list[np.ndarray]:
    """Filtert ungültige Frames aus einer Frame-Liste.

    Ungültige Frames sind: None, kein ndarray, falsches Shape,
    leere Arrays.

    Args:
        frames: Liste von potenziell ungültigen Frames.

    Returns:
        Liste von gültigen BGR numpy Arrays (H, W, 3).

    Beispiel:
        >>> valid = validate_frames([None, np.zeros((480, 640, 3), dtype=np.uint8)])
        >>> len(valid)
        1
    """
    valid: list[np.ndarray] = []
    for f in frames:
        if f is None:
            continue
        if not isinstance(f, np.ndarray):
            continue
        if f.ndim != 3 or f.shape[2] != 3:
            continue
        if f.size == 0:
            continue
        valid.append(f)
    return valid


def build_frame_sequence(
    frames: list,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
) -> dict:
    """Baut eine standardisierte Frame-Sequenz aus einer Rohliste.

    Validiert Frames, kürzt auf max_frames (neueste Frames bevorzugt)
    und gibt ein einheitliches Dictionary zurück.

    Args:
        frames:     Liste von potentiell ungültigen Frames.
        fps:        Aufnahme-FPS (Schätzung oder gemessen). Wird auf ≥ 1.0 geclippt.
        max_frames: Maximale Anzahl zu haltender Frames (Standard: 150).

    Returns:
        Dictionary mit:
            - frames (list[np.ndarray]): Validierte und ggf. gekürzte Frames.
            - fps (float):               Effektive FPS (≥ 1.0).
            - frame_count (int):         Anzahl gültiger Frames.

    Beispiel:
        >>> seq = build_frame_sequence([np.zeros((480, 640, 3), dtype=np.uint8)], fps=30.0)
        >>> seq["frame_count"]
        1
    """
    valid = validate_frames(frames)

    # Neueste max_frames behalten wenn zu viele Frames
    if len(valid) > max_frames:
        valid = valid[-max_frames:]

    effective_fps = max(1.0, float(fps))

    return {
        "frames": valid,
        "fps": round(effective_fps, 2),
        "frame_count": len(valid),
    }


def capture_frame_sequence(
    seconds: float = DEFAULT_CAPTURE_SECONDS,
    fps: float = DEFAULT_FPS,
    camera_index: int = 0,
) -> dict:
    """Nimmt eine Frame-Sequenz von der lokalen Webcam auf (OpenCV, kein Browser).

    Verwendet cv2.VideoCapture. Geeignet für Tests und Nicht-Browser-Umgebungen.
    In der Streamlit-App wird stattdessen streamlit-webrtc verwendet
    (siehe app/tests/fixation_test.py).

    HINWEISE:
        - Kamera muss verfügbar und freigegeben sein.
        - FPS aus CAP_PROP_FPS kann geräteabhängig von der tatsächlichen Rate abweichen.
        - Stabilitätsanalyse abhängig von Distanz zur Kamera.

    Args:
        seconds:       Aufnahmedauer in Sekunden (Standard: 3.0).
        fps:           Angestrebte FPS als Fallback wenn Kamera keinen Wert liefert.
        camera_index:  OpenCV-Kameraindex (0 = Standard-Webcam).

    Returns:
        Dictionary mit frames, fps, frame_count.
        Bei Fehler (keine Kamera): frames=[], fps=fps, frame_count=0.

    Beispiel:
        >>> seq = capture_frame_sequence(seconds=1.0)
        >>> isinstance(seq["frames"], list)
        True
    """
    try:
        import cv2  # Import hier: Modul ohne cv2 importierbar (für Tests)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {"frames": [], "fps": fps, "frame_count": 0}

        # Tatsächliche FPS der Kamera auslesen (Fallback: fps-Parameter)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 0:
            actual_fps = fps

        frames: list[np.ndarray] = []
        start = time.monotonic()
        deadline = start + max(0.1, float(seconds))
        max_f = min(MAX_FRAMES, int(actual_fps * seconds) + 5)

        while time.monotonic() < deadline and len(frames) < max_f:
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()

        return build_frame_sequence(frames, fps=actual_fps)

    except Exception:  # noqa: BLE001
        # Defensive: kein Crash wenn Kamera nicht verfügbar (z.B. CI-Umgebung)
        return {"frames": [], "fps": fps, "frame_count": 0}
