"""Augen-Metriken und Blink-Detection auf Basis von Face-Landmarks – VisionFit AI v0.2.

Berechnet den Eye Aspect Ratio (EAR) nach Soukupova & Cech (2016) und erkennt
Blinkereignisse in Frame-Sequenzen heuristisch.

WICHTIGE HINWEISE:
    - Dieses Modul ist KEIN Medizinprodukt und liefert KEINE klinische Diagnose.
    - Alle Schwellenwerte sind heuristische Näherungen ohne klinische Validierung.
    - Blink Detection ist erst mit Multi-Frame-Daten (≥ 30 Frames) belastbar
      aussagekräftig. Bei Einzelbildern ist is_reliable=False.
    - Der EAR-Wert variiert je nach Aufnahmewinkel, Beleuchtung und Person.

EAR-Formel:
    EAR = (d(p2,p6) + d(p3,p5)) / (2 · d(p1,p4))

    Wobei p1–p6 die 6 Augen-Landmark-Punkte sind (Außenwinkel, obere/untere Lider,
    Innenwinkel). Normalbereich offenes Auge: EAR ≈ 0.25–0.35. Blink: EAR < 0.20.

Augen-Indizes (MediaPipe FaceMesh, gilt für refine_landmarks=True und False):
    LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]
    Reihenfolge: [p1_außen, p2_oben_außen, p3_oben_innen, p4_innen, p5_unten_innen, p6_unten_außen]

Vorbereitet für v0.3 (Multi-Frame / Video):
    summarize_eye_metrics() ist vollständig für Echtzeit-Videostreams ausgelegt.
    In der aktuellen MVP-App (Einzelbild via st.camera_input) wird is_reliable=False
    zurückgegeben; die Funktion ist dennoch vollständig testbar.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from app.cv.face_mesh import get_landmark_xy

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

# MediaPipe FaceMesh Augen-Landmark-Indizes
# Reihenfolge: [p1_außen, p2_oben_außen, p3_oben_innen, p4_innen, p5_unten_innen, p6_unten_außen]
LEFT_EYE_INDICES: list[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES: list[int] = [33, 160, 158, 133, 153, 144]

# EAR-Schwellenwert für Blink-Erkennung (heuristisch, nicht klinisch validiert)
DEFAULT_BLINK_THRESHOLD: float = 0.20

# Minimale Frames unterhalb des Schwellenwerts für einen gültigen Blink
DEFAULT_MIN_BLINK_FRAMES: int = 2

# Mindest-Frames für zuverlässige Auswertung (≈ 1 Sekunde bei 30 fps)
MIN_RELIABLE_FRAMES: int = 30

# Normaler EAR-Bereich für offene Augen (Orientierungswert)
EAR_NORMAL_LOW: float = 0.20
EAR_NORMAL_HIGH: float = 0.40


class EyeMetricsSummary(NamedTuple):
    """Datenstruktur für zusammengefasste Augen-Metriken."""

    ear_mean: float          # Mittlerer EAR über alle Frames
    ear_std: float           # Standardabweichung des EAR
    blink_count: int         # Anzahl erkannter Blink-Ereignisse
    blink_rate: float        # Blinks pro Minute
    frame_count: int         # Anzahl gültiger Frames
    is_reliable: bool        # True wenn ≥ MIN_RELIABLE_FRAMES


# ──────────────────────────────────────────────────────────────────────────────
# Interne Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _euclidean(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Euklidischer Abstand zwischen zwei Pixel-Punkten."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _count_blink_events(
    ear_history: list[float],
    threshold: float = DEFAULT_BLINK_THRESHOLD,
    min_frames: int = DEFAULT_MIN_BLINK_FRAMES,
) -> int:
    """Zählt distinkte Blink-Ereignisse in einer EAR-Sequenz.

    Ein Blink-Ereignis = aufeinanderfolgende Frames mit EAR < threshold,
    mindestens min_frames lang.

    Args:
        ear_history:  EAR-Werte pro Frame.
        threshold:    EAR-Schwellenwert (Blink wenn EAR < threshold).
        min_frames:   Mindest-Frames für ein gültiges Blink-Ereignis.

    Returns:
        Anzahl distinctar Blink-Ereignisse (int ≥ 0).
    """
    if not ear_history:
        return 0

    events = 0
    run_length = 0
    event_counted = False

    for ear in ear_history:
        if ear < threshold:
            run_length += 1
            if run_length >= min_frames and not event_counted:
                events += 1
                event_counted = True
        else:
            run_length = 0
            event_counted = False

    return events


# ──────────────────────────────────────────────────────────────────────────────
# Öffentliche API
# ──────────────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(
    landmarks: list,
    eye_indices: list[int],
    image_shape: tuple,
) -> float:
    """Berechnet den Eye Aspect Ratio (EAR) für ein Auge.

    EAR = (d(p2,p6) + d(p3,p5)) / (2 · d(p1,p4))

    Wobei p1–p6 die Augen-Landmark-Punkte in folgender Reihenfolge sind:
    [p1_außen, p2_oben_außen, p3_oben_innen, p4_innen, p5_unten_innen, p6_unten_außen]

    HINWEIS: Dies ist eine heuristische Näherung. Der EAR variiert je nach
    Kamerawinkel, Beleuchtung und individueller Augenform. Schwellenwerte
    sind nicht klinisch validiert.

    Args:
        landmarks:    Liste von (x, y, z)-Tupeln in normierten Koordinaten [0.0, 1.0],
                      wie von detect_face_landmarks() zurückgegeben.
        eye_indices:  6 Landmark-Indizes in der Reihenfolge
                      [außen, oben_außen, oben_innen, innen, unten_innen, unten_außen].
        image_shape:  Bildgröße als (height, width, ...) für Pixelkonvertierung.

    Returns:
        EAR-Wert als float (typisch 0.25–0.35 bei offenem Auge, < 0.20 beim Blinzeln).
        0.0 bei fehlenden Landmarks oder Division-by-Zero.

    Raises:
        Kein Raise – fehlende Daten werden defensiv mit Rückgabe 0.0 behandelt.
    """
    if not landmarks or len(eye_indices) != 6:
        return 0.0

    # Defensive Prüfung: alle Indizes müssen gültig sein
    max_idx = max(eye_indices)
    if max_idx >= len(landmarks):
        return 0.0

    # Pixel-Koordinaten der 6 Augenpunkte
    try:
        p1, p2, p3, p4, p5, p6 = [
            get_landmark_xy(landmarks, idx, image_shape)
            for idx in eye_indices
        ]
    except (IndexError, TypeError, ValueError):
        return 0.0

    # Vertikale Abstände (obere/untere Lider)
    vert_1 = _euclidean(p2, p6)  # p2_oben_außen ↔ p6_unten_außen
    vert_2 = _euclidean(p3, p5)  # p3_oben_innen ↔ p5_unten_innen

    # Horizontaler Abstand (Augenbreite)
    horiz = _euclidean(p1, p4)

    if horiz < 1.0:  # Pixel-Abstand zu klein → kein valides Auge
        return 0.0

    ear = (vert_1 + vert_2) / (2.0 * horiz)
    return round(float(ear), 6)


def detect_blink(
    ear_history: list[float],
    threshold: float = DEFAULT_BLINK_THRESHOLD,
    min_frames: int = DEFAULT_MIN_BLINK_FRAMES,
) -> bool:
    """Prüft ob in einer EAR-Sequenz mindestens ein Blink stattgefunden hat.

    Ein Blink wird erkannt wenn der EAR-Wert für mindestens min_frames
    aufeinanderfolgende Frames unter threshold fällt.

    HINWEIS: Erst ab mehreren Frames (min. 10, besser 30+) belastbar.
    Mit einem einzelnen Frame ist das Ergebnis formal gültig aber nicht
    aussagekräftig – bei EAR < threshold und min_frames=1 wird True zurückgegeben.

    Args:
        ear_history:  Liste von EAR-Werten, ein Wert pro Frame.
        threshold:    EAR-Schwellenwert (Blink wenn darunter). Standard: 0.20.
        min_frames:   Mindest aufeinanderfolgende Frames unter threshold. Standard: 2.

    Returns:
        True wenn mindestens ein Blink erkannt wurde, sonst False.
        False bei leerer Eingabe.
    """
    return _count_blink_events(ear_history, threshold, min_frames) > 0


def blink_rate(
    blink_events: list[bool],
    fps: float,
) -> float:
    """Berechnet die Blinkrate in Blinks pro Minute.

    Zählt distincte Blink-Ereignisse (Übergänge False → True) und normiert
    auf eine Minute.

    Normaler Bereich für Erwachsene: 15–20 Blinks/Minute (Orientierungswert,
    stark kontextabhängig und individuell verschieden).

    Args:
        blink_events:  Pro-Frame Bool-Liste (True = Blink in diesem Frame).
        fps:           Frames pro Sekunde der Aufnahme.

    Returns:
        Blinkrate als float (Blinks/Minute). 0.0 bei leerer Eingabe oder fps ≤ 0.
    """
    if not blink_events or fps <= 0:
        return 0.0

    # Distinct Blink-Ereignisse = Anzahl steigender Flanken (False → True)
    event_count = 0
    prev = False
    for is_blink in blink_events:
        if is_blink and not prev:
            event_count += 1
        prev = is_blink

    n_frames = len(blink_events)
    duration_seconds = n_frames / fps

    if duration_seconds <= 0:
        return 0.0

    rate = event_count / duration_seconds * 60.0
    return round(float(rate), 1)


def label_blink_rate(rate: float) -> str:
    """Gibt eine kurze deutsche Interpretation der Blinkrate zurück.

    Orientierungswerte (nicht klinisch validiert):
    - Normal: 15–20 Blinks/Minute
    - Niedrig: < 10 (oft bei Bildschirmarbeit oder starker Konzentration)
    - Sehr hoch: > 40 (Irritation, Müdigkeit oder Reflexreaktion möglich)

    Args:
        rate:  Blinkrate in Blinks/Minute.

    Returns:
        Deutsches Label als String.
    """
    if rate <= 0.0:
        return "Kein Blinken erkannt"
    if rate < 5.0:
        return "Sehr selten"
    if rate < 10.0:
        return "Unterdurchschnittlich"
    if rate < 25.0:
        return "Normal"
    if rate < 40.0:
        return "Erhöht"
    return "Sehr häufig"


def summarize_eye_metrics(
    frames_landmarks: list[list],
    image_shape: tuple,
    fps: float = 30.0,
    blink_threshold: float = DEFAULT_BLINK_THRESHOLD,
    min_blink_frames: int = DEFAULT_MIN_BLINK_FRAMES,
) -> dict:
    """Berechnet alle Augen-Metriken aus einer Frame-Sequenz.

    Haupteinstiegspunkt für externe Aufrufer. Vollständig Streamlit-unabhängig.

    HINWEIS: Für belastbare Ergebnisse werden mindestens 30 Frames benötigt
    (≈ 1 Sekunde bei 30 fps). Bei weniger Frames ist is_reliable=False.
    In der aktuellen MVP-App (Einzelbild) ist diese Funktion vorbereitet aber
    noch nicht im UI integriert – das erfolgt in v0.3 (Multi-Frame).

    Args:
        frames_landmarks:  Liste von Landmark-Listen (eine pro Frame),
                           wie von detect_face_landmarks() geliefert.
                           Jede Liste enthält (x, y, z)-Tupel [0.0, 1.0].
        image_shape:       Bildgröße als (height, width, ...).
        fps:               Frames pro Sekunde (Standard: 30.0).
        blink_threshold:   EAR-Schwellenwert für Blink-Erkennung.
        min_blink_frames:  Mindest-Frames unter threshold für gültigen Blink.

    Returns:
        Dictionary mit:
            - ear_mean (float):       Mittlerer EAR beider Augen über alle Frames.
            - ear_std (float):        Standardabweichung des EAR.
            - blink_count (int):      Anzahl erkannter Blink-Ereignisse.
            - blink_rate (float):     Blinks pro Minute.
            - blink_events (list):    Pro-Frame Bool-Liste (True = Blink).
            - ear_history (list):     EAR-Werte pro Frame (für Visualisierung).
            - frame_count (int):      Anzahl ausgewerteter Frames.
            - is_reliable (bool):     True wenn ≥ MIN_RELIABLE_FRAMES Frames.
            - label (str):            Deutsche Gesamtbewertung.
            - warnung (str | None):   Deutsche Warnung bei unzuverlässigen Daten.
    """
    ear_history: list[float] = []

    for frame_lm in frames_landmarks:
        if not frame_lm:
            continue

        # EAR beider Augen berechnen, Durchschnitt nehmen
        ear_left = eye_aspect_ratio(frame_lm, LEFT_EYE_INDICES, image_shape)
        ear_right = eye_aspect_ratio(frame_lm, RIGHT_EYE_INDICES, image_shape)

        # Wenn beide Augen erkannt → Durchschnitt; wenn nur eines → dessen Wert
        if ear_left > 0.0 and ear_right > 0.0:
            ear_history.append((ear_left + ear_right) / 2.0)
        elif ear_left > 0.0:
            ear_history.append(ear_left)
        elif ear_right > 0.0:
            ear_history.append(ear_right)
        # Wenn beide 0.0 → Frame überspringen (keine validen Augen erkannt)

    n = len(ear_history)
    is_reliable = n >= MIN_RELIABLE_FRAMES

    # EAR-Statistiken
    if n >= 2:
        ear_mean = float(np.mean(ear_history))
        ear_std = float(np.std(ear_history, ddof=1))
    elif n == 1:
        ear_mean = ear_history[0]
        ear_std = 0.0
    else:
        ear_mean = 0.0
        ear_std = 0.0

    # Blink-Erkennung
    blink_count = _count_blink_events(ear_history, blink_threshold, min_blink_frames)

    # Pro-Frame Blink-Bool-Liste (für blink_rate-Funktion)
    per_frame_blinks = [e < blink_threshold for e in ear_history]
    calculated_rate = blink_rate(per_frame_blinks, fps)

    # Label ableiten
    if n == 0:
        label = "Keine Augen-Daten verfügbar"
    elif not is_reliable:
        label = "Unzureichende Datenbasis (Einzelbild)"
    else:
        label = label_blink_rate(calculated_rate)

    # Warnung generieren
    warnung: str | None = None
    if n == 0:
        warnung = (
            "Keine gültigen Augen-Landmarks gefunden. "
            "Bitte stellen Sie sicher, dass das Gesicht vollständig erkannt wurde."
        )
    elif not is_reliable:
        warnung = (
            f"Nur {n} Frame(s) ausgewertet. "
            f"Für zuverlässige Blink-Erkennung werden mindestens "
            f"{MIN_RELIABLE_FRAMES} Frames benötigt (≈ 1 Sekunde bei 30 fps). "
            "Diese Auswertung wird erst in v0.3 (Multi-Frame) vollständig unterstützt."
        )

    return {
        "ear_mean": round(ear_mean, 4),
        "ear_std": round(ear_std, 4),
        "blink_count": blink_count,
        "blink_rate": calculated_rate,
        "blink_events": per_frame_blinks,
        "ear_history": [round(e, 4) for e in ear_history],
        "frame_count": n,
        "is_reliable": is_reliable,
        "label": label,
        "warnung": warnung,
    }
