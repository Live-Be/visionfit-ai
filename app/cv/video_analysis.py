"""Video-basierte Analyse-Pipeline – VisionFit AI v0.3.

Orchestriert die vollständige Analyse einer Frame-Sequenz:
    1. Bildqualität (Helligkeit + Kontrast aus mittlerem Frame)
    2. Landmark-Extraktion (MediaPipe FaceMesh)
    3. Kopfstabilitäts-Analyse (Nasenspitzen-Tracking)
    4. Blink-Erkennung (EAR-basiert)

Das Modul ist vollständig Streamlit-unabhängig und pytest-testbar.
Alle CV-Teilmodule (landmark_pipeline, head_stability, eye_metrics) werden
aufgerufen und ihre Ergebnisse zu einem einheitlichen Ergebnis-Dictionary
zusammengeführt.

WICHTIGE HINWEISE (Limitationen):
    - Heuristische Analyse, kein Medizinprodukt, keine klinische Diagnose.
    - Bildqualität wird aus dem mittleren Frame berechnet; stark blurrige
      oder über-/unterbelichtete Aufnahmen beeinflussen das Ergebnis.
    - Blinkrate abhängig von Kamera-FPS, Beleuchtung und individuellem Blinkverhalten.
    - Stabilitätsanalyse abhängig von Distanz zur Kamera und Aufnahmewinkel.
    - face_detection_rate < 0.3 signalisiert unzuverlässige Gesamtanalyse.
    - Für klinisch relevante Auswertungen sind kalibrierte Geräte erforderlich.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.cv.landmark_pipeline import extract_landmarks_from_frames
from app.cv.head_stability import summarize_head_stability
from app.cv.eye_metrics import summarize_eye_metrics


# Mindest-Gesichtserkennungsrate für eine belastbare Auswertung
MIN_RELIABLE_DETECTION_RATE: float = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Interne Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _image_quality_from_frame(frame: np.ndarray) -> tuple[float, float]:
    """Berechnet Helligkeit und Kontrast eines BGR-Frames.

    Args:
        frame: BGR numpy Array (H, W, 3).

    Returns:
        Tuple (brightness, contrast) als float.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)), float(np.std(gray))


def _empty_analysis_result() -> dict:
    """Gibt ein leeres Analyse-Ergebnis zurück (bei 0 Frames)."""
    return {
        "brightness": 0.0,
        "contrast": 0.0,
        "face_detection_rate": 0.0,
        "head_stability_score": 0.0,
        "head_stability_label": "Keine Daten",
        "head_stability_reliable": False,
        "blink_rate": 0.0,
        "blink_count": 0,
        "ear_mean": 0.0,
        "ear_std": 0.0,
        "blink_reliable": False,
        "frame_count": 0,
        "is_reliable": False,
        "warnung": "Keine Frames für Analyse vorhanden.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Öffentliche API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_video_sequence(
    frames: list[np.ndarray],
    fps: float = 30.0,
) -> dict:
    """Analysiert eine Video-Frame-Sequenz vollständig.

    Workflow:
        1. Bildqualität aus mittlerem Frame (robuster als erster Frame,
           da Kamera-Autofokus nach dem ersten Frame oft stabiler ist).
        2. Landmark-Extraktion für alle Frames via MediaPipe.
        3. Kopfstabilitäts-Analyse aus Nasenspitzen-Positionen.
        4. Blink-Erkennung aus EAR-Werten.

    HINWEIS: face_detection_rate < MIN_RELIABLE_DETECTION_RATE (0.3) bedeutet,
    dass weniger als 30% der Frames ein Gesicht enthielten. In diesem Fall
    sollte score_fixation_no_face() statt score_fixation_combined() verwendet werden.

    Args:
        frames: Liste von BGR-Frames als numpy Arrays.
                Leere Liste gibt ein "leeres" Ergebnis-Dictionary zurück.
        fps:    Frames pro Sekunde der Aufnahme (Standard: 30.0).
                Beeinflusst die Blinkraten-Berechnung.

    Returns:
        Dictionary mit:
            - brightness (float):              Mittlere Helligkeit (0–255).
            - contrast (float):                Kontrast als Std-Abw. Pixelwerte.
            - face_detection_rate (float):     Anteil Frames mit erkanntem Gesicht (0–1).
            - head_stability_score (float):    Stabilitätsscore (0–100).
            - head_stability_label (str):      Deutsches Stabilitäts-Label.
            - head_stability_reliable (bool):  Zuverlässig wenn ≥ 2 Frames mit Gesicht.
            - blink_rate (float):              Blinks pro Minute.
            - blink_count (int):               Anzahl erkannter Blink-Ereignisse.
            - ear_mean (float):                Mittlerer EAR beider Augen.
            - ear_std (float):                 Standardabweichung EAR.
            - blink_reliable (bool):           Zuverlässig wenn ≥ 30 Frames mit Augen.
            - frame_count (int):               Anzahl analysierter Frames.
            - is_reliable (bool):              True wenn face_detection_rate ≥ 0.3.
            - warnung (str | None):            Deutsche Warnung bei unzuverlässigen Daten.

    Beispiel:
        >>> from app.cv.video_analysis import analyze_video_sequence
        >>> black_frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 10
        >>> result = analyze_video_sequence(black_frames, fps=30.0)
        >>> result["face_detection_rate"]
        0.0
    """
    if not frames:
        return _empty_analysis_result()

    # Mittlerer Frame für Bildqualitätsanalyse (robuster als Frame 0)
    mid_idx = len(frames) // 2
    brightness, contrast = _image_quality_from_frame(frames[mid_idx])

    # Landmark-Extraktion für alle Frames
    pipeline = extract_landmarks_from_frames(frames)
    frames_landmarks = pipeline["frames_landmarks"]
    face_detection_rate = pipeline["face_detection_rate"]

    # Head Stability (nutzt normierte Koordinaten, bildrößenunabhängig)
    stability = summarize_head_stability(frames_landmarks)

    # Blink Detection (nutzt Pixel-Koordinaten via get_landmark_xy)
    image_shape = frames[mid_idx].shape
    blinks = summarize_eye_metrics(frames_landmarks, image_shape, fps=fps)

    # Gesamtzuverlässigkeit
    is_reliable = face_detection_rate >= MIN_RELIABLE_DETECTION_RATE
    warnung: str | None = None
    if not is_reliable:
        warnung = (
            f"Gesicht nur in {face_detection_rate * 100:.0f}% der Frames erkannt "
            f"(mindestens {int(MIN_RELIABLE_DETECTION_RATE * 100)}% erforderlich). "
            "Bitte stellen Sie sicher, dass Ihr Gesicht vollständig und gut beleuchtet "
            "sichtbar ist."
        )

    return {
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "face_detection_rate": face_detection_rate,
        "head_stability_score": stability["head_stability_score"],
        "head_stability_label": stability["label"],
        "head_stability_reliable": stability["is_reliable"],
        "blink_rate": blinks["blink_rate"],
        "blink_count": blinks["blink_count"],
        "ear_mean": blinks["ear_mean"],
        "ear_std": blinks["ear_std"],
        "blink_reliable": blinks["is_reliable"],
        "frame_count": len(frames),
        "is_reliable": is_reliable,
        "warnung": warnung,
    }
