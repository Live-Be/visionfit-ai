"""Kopfstabilitäts-Analyse auf Basis von Face-Landmark-Sequenzen – VisionFit AI v0.2.

Berechnet aus einer Folge von Landmark-Positionen (mehrere Frames),
wie stabil der Kopf einer Person während der Aufnahme war.

HINWEIS: Für aussagekräftige Ergebnisse werden mindestens 10–30 Frames benötigt
(z.B. 1–3 Sekunden Videoaufnahme bei 10 fps). Bei Einzelbildern ist der Score
formal gültig (0 Bewegung = perfekte Stabilität), aber nicht informativ.
Dieses Modul ist vollständig Streamlit-unabhängig und pytest-testbar.

Referenz-Landmark:
    Index 1 = Nasenspitze (MediaPipe FaceMesh, konstant und robust).
    Normalisierte Koordinaten (0.0–1.0) werden bevorzugt, da sie
    bildgrößenunabhängig sind.

Vorbereitet für Phase 3 (Blink Detection):
    get_landmark_xy() aus face_mesh.py kann ebenfalls verwendet werden,
    um Augen-Landmarks pixelgenau zu extrahieren.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# Nasenspitzen-Index im MediaPipe FaceMesh (refine_landmarks=True oder False)
NOSE_TIP_INDEX: int = 1

# Referenz-Maximum für combined_motion_std (normalisierte Einheiten).
# Wert 0.05 entspricht ~5 % der Bildbreite/-höhe – ab hier als instabil bewertet.
DEFAULT_MAX_MOTION_STD: float = 0.05


class StabilityMetrics(NamedTuple):
    """Datenstruktur für berechnete Stabilitätsmetriken."""

    std_x: float               # Standardabweichung der x-Positionen
    std_y: float               # Standardabweichung der y-Positionen
    combined_motion_std: float # Euklidische Kombination: sqrt(std_x² + std_y²)
    frame_count: int           # Anzahl ausgewerteter Frames
    is_reliable: bool          # True wenn >= 2 Frames vorhanden


# ──────────────────────────────────────────────────────────────────────────────
# Kern-Funktionen (rein, ohne Seiteneffekte)
# ──────────────────────────────────────────────────────────────────────────────

def extract_reference_point_sequence(
    frames_landmarks: list[list],
    landmark_index: int = NOSE_TIP_INDEX,
) -> list[tuple[float, float]]:
    """Extrahiert eine (x, y)-Positionssequenz aus mehreren Frame-Landmark-Listen.

    Überspringt Frames in denen das Landmark nicht vorhanden ist (defensiv).

    Args:
        frames_landmarks:  Liste von Landmark-Listen, eine pro Frame.
                           Jede Liste enthält (x, y, z)-Tupel in normierten
                           Koordinaten [0.0, 1.0], wie von detect_face_landmarks()
                           zurückgegeben.
        landmark_index:    Index des Referenz-Landmarks (Standard: 1 = Nasenspitze).

    Returns:
        Liste von (x, y)-Float-Tupeln für jeden gültigen Frame.
        Leere Liste wenn frames_landmarks leer ist oder kein Landmark gefunden wurde.

    Beispiel:
        >>> frames = [[(0.5, 0.4, 0.0), ...], [(0.51, 0.41, 0.0), ...]]
        >>> extract_reference_point_sequence(frames)
        [(0.5, 0.4), (0.51, 0.41)]
    """
    positions: list[tuple[float, float]] = []

    for frame_lm in frames_landmarks:
        if not frame_lm:
            continue
        if landmark_index >= len(frame_lm):
            continue
        x, y, *_ = frame_lm[landmark_index]
        positions.append((float(x), float(y)))

    return positions


def calculate_position_std(
    positions: list[tuple[float, float]],
) -> StabilityMetrics:
    """Berechnet die Positionsstreuung aus einer (x, y)-Sequenz.

    Args:
        positions:  Liste von (x, y)-Tupeln in normierten Koordinaten.

    Returns:
        StabilityMetrics mit std_x, std_y, combined_motion_std,
        frame_count und is_reliable.

    Hinweis:
        Bei 0 oder 1 Frame ist is_reliable=False und combined_motion_std=0.0.
        Das ist kein Fehler, sondern dokumentiertes defensives Verhalten.
    """
    n = len(positions)

    if n < 2:
        return StabilityMetrics(
            std_x=0.0,
            std_y=0.0,
            combined_motion_std=0.0,
            frame_count=n,
            is_reliable=False,
        )

    xs = np.array([p[0] for p in positions], dtype=float)
    ys = np.array([p[1] for p in positions], dtype=float)

    std_x = float(np.std(xs, ddof=1))  # Stichproben-Std (ddof=1)
    std_y = float(np.std(ys, ddof=1))

    # Euklidische Kombination: Magnitude der 2D-Bewegungsstreuung
    combined = math.sqrt(std_x ** 2 + std_y ** 2)

    return StabilityMetrics(
        std_x=round(std_x, 6),
        std_y=round(std_y, 6),
        combined_motion_std=round(combined, 6),
        frame_count=n,
        is_reliable=True,
    )


def calculate_head_stability_score(
    combined_motion_std: float,
    max_motion_std: float = DEFAULT_MAX_MOTION_STD,
) -> float:
    """Konvertiert combined_motion_std in einen Stabilitätsscore (0–100).

    Lineare Abnahme: 0 Bewegung → 100, max_motion_std Bewegung → 0.
    Werte über max_motion_std werden auf 0 geclippt.

    Args:
        combined_motion_std:  Euklidische Bewegungsstreuung (normiert).
        max_motion_std:       Referenz-Maximum (Standard: 0.05).

    Returns:
        Score als float im Bereich [0.0, 100.0].
    """
    if max_motion_std <= 0:
        return 100.0 if combined_motion_std == 0.0 else 0.0

    raw = (1.0 - combined_motion_std / max_motion_std) * 100.0
    return round(float(max(0.0, min(100.0, raw))), 1)


def label_head_stability(score: float) -> str:
    """Gibt eine kurze deutsche Interpretation des Stabilitäts-Scores zurück.

    Args:
        score:  Stabilitätsscore (0–100).

    Returns:
        Deutsches Label als String.
    """
    if score >= 80:
        return "Sehr stabil"
    if score >= 60:
        return "Stabil"
    if score >= 40:
        return "Mäßig stabil"
    if score >= 20:
        return "Instabil"
    return "Sehr instabil"


# ──────────────────────────────────────────────────────────────────────────────
# High-Level Zusammenfassung
# ──────────────────────────────────────────────────────────────────────────────

def summarize_head_stability(
    frames_landmarks: list[list],
    landmark_index: int = NOSE_TIP_INDEX,
    max_motion_std: float = DEFAULT_MAX_MOTION_STD,
) -> dict:
    """Berechnet alle Kopfstabilitätsmetriken aus einer Frame-Sequenz.

    Haupteinstiegspunkt für externe Aufrufer.

    Args:
        frames_landmarks:  Liste von Landmark-Listen (eine pro Frame),
                           wie von detect_face_landmarks() geliefert.
        landmark_index:    Referenz-Landmark-Index (Standard: 1 = Nasenspitze).
        max_motion_std:    Normierungsgrenze für Score-Berechnung.

    Returns:
        Dictionary mit:
            - head_stability_score (float):  Score 0–100.
            - label (str):                   Deutsche Interpretation.
            - std_x (float):                 Streuung x-Achse.
            - std_y (float):                 Streuung y-Achse.
            - combined_motion_std (float):   Gesamtbewegungsstreuung.
            - frame_count (int):             Ausgewertete Frames.
            - is_reliable (bool):            Ob genug Frames für Auswertung.
            - landmark_index (int):          Verwendeter Landmark-Index.
            - warnung (str | None):          Deutsche Warnung bei unzuverlässigen Daten.

    Beispiel:
        >>> lm_seq = [frame_result["landmarks"] for frame_result in detect_results]
        >>> summary = summarize_head_stability(lm_seq)
        >>> print(summary["head_stability_score"], summary["label"])
    """
    # Schritt 1: Positionen extrahieren
    positions = extract_reference_point_sequence(frames_landmarks, landmark_index)

    # Schritt 2: Streuung berechnen
    metrics = calculate_position_std(positions)

    # Schritt 3: Score berechnen
    score = calculate_head_stability_score(metrics.combined_motion_std, max_motion_std)

    # Schritt 4: Label ableiten
    label = label_head_stability(score)

    # Schritt 5: Warnung wenn Daten nicht zuverlässig
    warnung: str | None = None
    if not metrics.is_reliable:
        if metrics.frame_count == 0:
            warnung = (
                "Keine Landmark-Daten vorhanden. "
                "Bitte stellen Sie sicher, dass das Gesicht erkannt wurde."
            )
        else:
            warnung = (
                f"Nur {metrics.frame_count} Frame(s) analysiert. "
                "Für zuverlässige Ergebnisse werden mindestens 10 Frames benötigt."
            )

    return {
        "head_stability_score": score,
        "label": label,
        "std_x": metrics.std_x,
        "std_y": metrics.std_y,
        "combined_motion_std": metrics.combined_motion_std,
        "frame_count": metrics.frame_count,
        "is_reliable": metrics.is_reliable,
        "landmark_index": landmark_index,
        "warnung": warnung,
    }
