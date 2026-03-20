"""Sakkadentest-Analyse – VisionFit AI.

Extrahiert horizontale Iris-Positionen aus Frame-Sequenzen (mit Timestamps)
und berechnet Sakkadenmetriken:
    - saccade_latency_ms      (Reaktionszeit nach Stimuluswechsel)
    - accuracy_score          (Anteil korrekt detektierter Blicksprünge)
    - correction_saccades_count (Nachkorrekturen nach initialem Landing)
    - symmetry_score          (Vergleich links→rechts vs rechts→links)

Iris-Tracking nutzt MediaPipe FaceMesh mit refine_landmarks=True,
das bereits in der restlichen App-Pipeline so initialisiert wird.

WICHTIGE HINWEISE:
    - Heuristische Analyse, kein Medizinprodukt, keine klinische Diagnose.
    - Iris-Tracking benötigt gute Beleuchtung und frontale Kameraposition.
    - Latenzwerte sind Schätzungen – kamera-FPS und Webcam-Latenz
      beeinflussen die Genauigkeit.
    - Dieses Modul ist vollständig Streamlit-unabhängig und pytest-testbar.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

# MediaPipe FaceMesh Iris-Landmark-Indizes (nur mit refine_landmarks=True, 478 Landmarks)
LEFT_IRIS_CENTER: int = 468
RIGHT_IRIS_CENTER: int = 473

# Mindest-Erkennungsrate für Iris-Tracking
MIN_IRIS_DETECTION_RATE: float = 0.4

# Physiologisch plausibles Latenzfenster (ms)
_LATENCY_MIN_MS: float = 60.0   # unter 60ms = antizipatorisch, kein echter Reflex
_LATENCY_MAX_MS: float = 600.0  # über 600ms = zu langsam, wohl kein Sakkadenreflex

# Suchfenster nach Stimuluswechsel für Sakkadenbeginn (Sekunden)
_SEARCH_WINDOW_S: float = 0.6

# Iris-Bewegungs-Schwellenwert für Sakkadenerkennung (normierte Einheiten / Frame)
_VELOCITY_THRESHOLD: float = 0.006

# EMA-Glättungsfaktor (niedrig = glatter, hoher = reaktiver)
_EMA_ALPHA: float = 0.35


# ──────────────────────────────────────────────────────────────────────────────
# Interne Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _extract_iris_x(landmarks: list) -> float | None:
    """Gibt die normierte horizontale Iris-Mittenposition zurück.

    Mittelt die x-Koordinaten beider Iris-Center (Landmark 468 und 473).
    Gibt None zurück wenn die Iris-Landmarks fehlen (< 474 Landmarks insgesamt,
    z. B. wenn refine_landmarks=False verwendet wurde).

    Args:
        landmarks: Liste von (x, y, z)-Tupeln in normierten Koordinaten [0.0, 1.0],
                   wie von detect_face_landmarks() geliefert.

    Returns:
        Normierter x-Wert [0.0, 1.0] oder None.
    """
    if not landmarks or len(landmarks) <= RIGHT_IRIS_CENTER:
        return None
    left_x = float(landmarks[LEFT_IRIS_CENTER][0])
    right_x = float(landmarks[RIGHT_IRIS_CENTER][0])
    return (left_x + right_x) / 2.0


def _ema_smooth(
    signal: list[float | None],
    alpha: float = _EMA_ALPHA,
) -> list[float | None]:
    """Exponential Moving Average Glättung mit Lücken-Interpolation.

    Lücken (None) werden mit dem letzten bekannten Wert gefüllt.
    Am Anfang bleiben Lücken None bis der erste Wert erscheint.

    Args:
        signal: Liste von float-Werten oder None.
        alpha:  Glättungsfaktor (0 = sehr glatt, 1 = kein Glätten).

    Returns:
        Geglättetes Signal gleicher Länge.
    """
    result: list[float | None] = []
    last: float | None = None
    for v in signal:
        if v is None:
            result.append(last)
        else:
            last = alpha * v + (1.0 - alpha) * last if last is not None else v
            result.append(last)
    return result


def _find_saccade_onset(
    gaze_x: list[float | None],
    frame_times: list[float],
    event_time: float,
    search_window_s: float = _SEARCH_WINDOW_S,
    velocity_threshold: float = _VELOCITY_THRESHOLD,
) -> float | None:
    """Sucht den Beginn einer Sakkade (Iris-Geschwindigkeitsspitze).

    Wertet die Iris-Geschwindigkeit frame-weise im Suchfenster aus.
    Gibt den Zeitpunkt des ersten Überschreitens des Schwellenwerts zurück
    (richtungsunabhängig, da Webcam-Spiegelung und individuelle Baseline variieren).

    Args:
        gaze_x:           Geglättetes Iris-x-Signal.
        frame_times:      Absolute Timestamps der Frames (Sekunden).
        event_time:       Zeitpunkt des Stimuluswechsels.
        search_window_s:  Maximales Suchfenster nach event_time.
        velocity_threshold: Mindest-Geschwindigkeit für Sakkadenbeginn.

    Returns:
        Absoluter Zeitpunkt des Sakkaden-Onset oder None.
    """
    # Basiswert: letzter Gaze-Wert exakt vor dem Ereignis (als Startpunkt)
    baseline: float | None = None
    for i in range(len(frame_times)):
        if frame_times[i] < event_time and gaze_x[i] is not None:
            baseline = float(gaze_x[i])

    prev_val = baseline
    for i in range(len(frame_times)):
        t = frame_times[i]
        if t < event_time:
            continue
        if t > event_time + search_window_s:
            break
        if gaze_x[i] is None:
            continue
        curr_val = float(gaze_x[i])
        if prev_val is not None:
            frame_velocity = abs(curr_val - prev_val)
            if frame_velocity > velocity_threshold:
                return t
        prev_val = curr_val
    return None


def _count_correction_saccades(
    gaze_x: list[float | None],
    frame_times: list[float],
    landing_start: float,
    landing_end: float,
    min_amplitude: float = 0.008,
) -> int:
    """Zählt Korrektursakkaden im Landing-Fenster.

    Eine Korrektursakkade = Richtungswechsel der Iris-Bewegung mit
    ausreichender Amplitude. Richtungswechsel werden als Vorzeichenwechsel
    der Frame-zu-Frame Differenz erkannt.

    Args:
        gaze_x:        Geglättetes Iris-x-Signal.
        frame_times:   Timestamps der Frames.
        landing_start: Beginn des Auswertungsfensters.
        landing_end:   Ende des Auswertungsfensters.
        min_amplitude: Mindest-Amplitude für einen gültigen Richtungswechsel.

    Returns:
        Anzahl erkannter Korrektursakkaden (int >= 0).
    """
    # Frames im Landing-Fenster sammeln
    window_vals: list[float] = []
    for i, t in enumerate(frame_times):
        if landing_start <= t <= landing_end and gaze_x[i] is not None:
            window_vals.append(float(gaze_x[i]))

    if len(window_vals) < 3:
        return 0

    diffs = [window_vals[k + 1] - window_vals[k] for k in range(len(window_vals) - 1)]

    corrections = 0
    for k in range(len(diffs) - 1):
        # Richtungswechsel mit ausreichender Amplitude auf beiden Seiten
        if (
            abs(diffs[k]) > min_amplitude
            and abs(diffs[k + 1]) > min_amplitude
            and ((diffs[k] > 0) != (diffs[k + 1] > 0))
        ):
            corrections += 1

    return corrections


# ──────────────────────────────────────────────────────────────────────────────
# Öffentliche API
# ──────────────────────────────────────────────────────────────────────────────

def extract_iris_x_sequence(
    frames: list[np.ndarray],
) -> dict:
    """Extrahiert die horizontale Iris-Position aus einer Frame-Sequenz.

    Haupteinstiegspunkt für die Iris-Extraktion. Streamlit-unabhängig.

    Args:
        frames: Liste von BGR-Frames (numpy Arrays, H x W x 3).

    Returns:
        Dictionary mit:
            - iris_x_raw (list[float|None]): Normierter x-Wert pro Frame.
            - iris_x_smooth (list[float|None]): EMA-geglättet.
            - face_detection_rate (float): Anteil Frames mit erkanntem Gesicht.
            - iris_detection_rate (float): Anteil Frames mit erkannter Iris.
            - frames_landmarks (list): Landmarks pro Frame.
    """
    if not frames:
        return {
            "iris_x_raw": [],
            "iris_x_smooth": [],
            "face_detection_rate": 0.0,
            "iris_detection_rate": 0.0,
            "frames_landmarks": [],
        }

    # Lazy import: cv2/mediapipe erst bei tatsächlicher Nutzung laden
    from app.cv.landmark_pipeline import extract_landmarks_from_frames  # noqa: PLC0415
    pipeline = extract_landmarks_from_frames(frames)
    frames_landmarks = pipeline["frames_landmarks"]
    face_detection_rate = pipeline["face_detection_rate"]

    iris_x_raw = [_extract_iris_x(lm) for lm in frames_landmarks]
    iris_x_smooth = _ema_smooth(iris_x_raw)

    valid_iris = sum(1 for x in iris_x_raw if x is not None)
    iris_detection_rate = round(valid_iris / len(iris_x_raw), 3) if iris_x_raw else 0.0

    return {
        "iris_x_raw": iris_x_raw,
        "iris_x_smooth": iris_x_smooth,
        "face_detection_rate": face_detection_rate,
        "iris_detection_rate": iris_detection_rate,
        "frames_landmarks": frames_landmarks,
    }


def analyze_saccade_test(
    timed_frames: list[tuple[np.ndarray, float]],
    stimulus_events: list[dict],
) -> dict:
    """Analysiert einen Sakkadentest-Durchlauf vollständig.

    Verarbeitet aufgenommene Frames (mit Timestamps) und die bekannte
    Stimulus-Ereignissequenz. Berechnet alle MVP-Metriken.

    Args:
        timed_frames:    Liste von (frame_bgr, timestamp_s) Tuples.
                         timestamp_s = absolute Unix-Zeit bei Frame-Aufnahme.
        stimulus_events: Liste von Dictionaries:
                         {'time': float, 'direction': 'left'|'right'}
                         Reihenfolge: chronologisch, 'time' = absoluter Unix-Timestamp.

    Returns:
        Dictionary mit:
            - latency_ms_mean (float|None):     Mittlere Sakkaden-Latenz in ms.
            - latency_ms_std (float|None):      Std der Latenz.
            - accuracy_score (float):           Erkennungsquote 0–100.
            - correction_saccades_count (int):  Korrekturbewegungen insgesamt.
            - symmetry_score (float|None):      Links/Rechts-Symmetrie 0–100.
            - quality_score (float):            Datenqualität 0–100.
            - interpretation_text (str):        Lesbare Kurzinterpretation.
            - raw_event_count (int):            Anzahl Stimulus-Events.
            - analyzed_event_count (int):       Auswertbare Events.
            - face_detection_rate (float):      Gesichtserkennungsrate.
            - iris_detection_rate (float):      Iris-Erkennungsrate.
            - head_movement_warning (bool):     Hinweis auf starke Kopfbewegung.
            - is_reliable (bool):               Zuverlässigkeit der Auswertung.
            - warnung (str|None):               Fehlermeldung bei schlechten Daten.
    """
    if not timed_frames:
        return _empty_saccade_result("Keine Frames aufgenommen.")

    if not stimulus_events:
        return _empty_saccade_result("Keine Stimulus-Events vorhanden.")

    frames = [f for f, _ in timed_frames]
    frame_times = [t for _, t in timed_frames]

    # Iris-Extraktion
    extraction = extract_iris_x_sequence(frames)
    face_detection_rate = extraction["face_detection_rate"]
    iris_detection_rate = extraction["iris_detection_rate"]
    frames_landmarks = extraction["frames_landmarks"]
    gaze_x = extraction["iris_x_smooth"]

    if iris_detection_rate < MIN_IRIS_DETECTION_RATE:
        warnung = (
            f"Iris nur in {iris_detection_rate * 100:.0f}% der Frames erkannt "
            f"(Minimum: {int(MIN_IRIS_DETECTION_RATE * 100)}%). "
            "Bitte auf gute Beleuchtung, frontale Kameraposition und "
            "ausreichend Abstand (40–70 cm) achten."
        )
        return _empty_saccade_result(warnung)

    # Pro Stimulus-Event analysieren
    latencies_ms: list[float] = []
    accuracies: list[float] = []
    corrections_per_event: list[int] = []
    left_latencies: list[float] = []
    right_latencies: list[float] = []

    for i, event in enumerate(stimulus_events):
        event_time = float(event["time"])
        direction = event["direction"]

        # Suchfenster: bis zum nächsten Event (oder +0.6s)
        if i + 1 < len(stimulus_events):
            next_event_time = float(stimulus_events[i + 1]["time"])
            window_end = min(event_time + _SEARCH_WINDOW_S, next_event_time - 0.1)
        else:
            window_end = event_time + _SEARCH_WINDOW_S

        # Mindestens 2 Frames im Fenster nötig
        frames_in_window = [j for j, t in enumerate(frame_times) if event_time <= t <= window_end]
        if len(frames_in_window) < 2:
            continue

        # Sakkaden-Onset detektieren
        onset_time = _find_saccade_onset(gaze_x, frame_times, event_time, _SEARCH_WINDOW_S)

        if onset_time is not None:
            latency_ms = (onset_time - event_time) * 1000.0
            if _LATENCY_MIN_MS <= latency_ms <= _LATENCY_MAX_MS:
                latencies_ms.append(latency_ms)
                if direction == "left":
                    left_latencies.append(latency_ms)
                else:
                    right_latencies.append(latency_ms)
                accuracies.append(1.0)  # Sakkade erkannt = korrekte Reaktion
            else:
                # Außerhalb physiol. Fenster: antizipatorisch oder kein Reflex
                accuracies.append(0.0)
        else:
            accuracies.append(0.0)

        # Korrektursakkaden: Fenster 250–800ms nach Stimuluswechsel
        corrections = _count_correction_saccades(
            gaze_x,
            frame_times,
            landing_start=event_time + 0.25,
            landing_end=event_time + 0.8,
        )
        corrections_per_event.append(corrections)

    n_events = len(stimulus_events)
    n_analyzed = len(latencies_ms)
    total_corrections = sum(corrections_per_event)

    # Metriken aggregieren
    latency_mean: float | None = float(np.mean(latencies_ms)) if latencies_ms else None
    latency_std: float | None = (
        float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) >= 2 else None
    )
    accuracy_score = float(np.mean(accuracies)) * 100.0 if accuracies else 0.0

    # Symmetrie: Latenzunterschied links vs. rechts (0=perfekt, 100=identisch)
    symmetry_score: float | None = None
    if left_latencies and right_latencies:
        mean_left = float(np.mean(left_latencies))
        mean_right = float(np.mean(right_latencies))
        diff_ms = abs(mean_left - mean_right)
        # 200ms Differenz → Score=0; 0ms → Score=100
        symmetry_score = max(0.0, 100.0 - diff_ms / 2.0)

    # Qualitätsscore: Coverage (auswertbare Events) × Iris-Erkennungsrate
    coverage = n_analyzed / max(n_events * 0.5, 1.0)
    quality_score = min(100.0, float(coverage) * iris_detection_rate * 100.0)

    # Kopfbewegungswarnung via bestehender Head-Stability-Logik
    head_movement_warning = _compute_head_warning(frames_landmarks)

    # Interpretation
    interpretation = _interpret_results(
        latency_mean, accuracy_score, total_corrections, symmetry_score, n_analyzed
    )

    is_reliable = quality_score >= 40.0 and n_analyzed >= 4

    return {
        "latency_ms_mean": round(latency_mean, 1) if latency_mean is not None else None,
        "latency_ms_std": round(latency_std, 1) if latency_std is not None else None,
        "accuracy_score": round(accuracy_score, 1),
        "correction_saccades_count": total_corrections,
        "symmetry_score": round(symmetry_score, 1) if symmetry_score is not None else None,
        "quality_score": round(quality_score, 1),
        "interpretation_text": interpretation,
        "raw_event_count": n_events,
        "analyzed_event_count": n_analyzed,
        "face_detection_rate": face_detection_rate,
        "iris_detection_rate": iris_detection_rate,
        "head_movement_warning": head_movement_warning,
        "is_reliable": is_reliable,
        "warnung": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Interne Auswertungs-Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _compute_head_warning(frames_landmarks: list) -> bool:
    """True wenn signifikante Kopfbewegung während des Tests erkannt wurde.

    Nutzt die bestehende head_stability-Logik (Nasenspitzen-Tracking).
    Gibt bei Analysefehlern False zurück (fail-safe).
    """
    try:
        from app.cv.head_stability import summarize_head_stability  # noqa: PLC0415
        stability = summarize_head_stability(frames_landmarks)
        return stability["head_stability_score"] < 50.0
    except Exception:  # noqa: BLE001
        return False


def _interpret_results(
    latency_ms: float | None,
    accuracy_score: float,
    corrections: int,
    symmetry_score: float | None,
    n_analyzed: int,
) -> str:
    """Erzeugt eine kurze, nicht-medizinische Interpretation der Sakkadenergebnisse.

    Gibt Hinweise in einfacher Sprache zurück – ohne medizinische Diagnose.
    """
    if n_analyzed < 3:
        return "Zu wenige auswertbare Sakkaden für eine Interpretation."

    parts: list[str] = []

    # Reaktionszeit
    if latency_ms is None:
        parts.append("Reaktionszeit: nicht messbar")
    elif latency_ms < 180:
        parts.append("Reaktionszeit: schnell")
    elif latency_ms < 280:
        parts.append("Reaktionszeit: normal")
    elif latency_ms < 380:
        parts.append("Reaktionszeit: leicht verzögert")
    else:
        parts.append("Reaktionszeit: verzögert")

    # Zielgenauigkeit
    if accuracy_score >= 80:
        parts.append("Zielgenauigkeit: gut")
    elif accuracy_score >= 55:
        parts.append("Zielgenauigkeit: mittel")
    else:
        parts.append("Zielgenauigkeit: niedrig")

    # Korrekturbewegungen
    if corrections == 0:
        parts.append("Korrekturbewegungen: keine")
    elif corrections <= 2:
        parts.append("Korrekturbewegungen: wenige")
    elif corrections <= 5:
        parts.append("Korrekturbewegungen: mehrere")
    else:
        parts.append("Korrekturbewegungen: viele")

    # Symmetrie
    if symmetry_score is not None:
        if symmetry_score >= 80:
            parts.append("Links/Rechts: ausgeglichen")
        elif symmetry_score >= 55:
            parts.append("Links/Rechts: leicht asymmetrisch")
        else:
            parts.append("Links/Rechts: asymmetrisch")

    return " · ".join(parts)


def _empty_saccade_result(warnung: str) -> dict:
    """Gibt ein leeres Ergebnis-Dictionary mit Fehlermeldung zurück."""
    return {
        "latency_ms_mean": None,
        "latency_ms_std": None,
        "accuracy_score": 0.0,
        "correction_saccades_count": 0,
        "symmetry_score": None,
        "quality_score": 0.0,
        "interpretation_text": "Test nicht auswertbar.",
        "raw_event_count": 0,
        "analyzed_event_count": 0,
        "face_detection_rate": 0.0,
        "iris_detection_rate": 0.0,
        "head_movement_warning": False,
        "is_reliable": False,
        "warnung": warnung,
    }
