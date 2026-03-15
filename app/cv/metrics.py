"""Bildmetriken für Helligkeits- und Kontraststärke."""

import numpy as np


def calculate_brightness_score(gray: np.ndarray) -> float:
    """Berechnet den mittleren Helligkeitswert (0–255).

    Args:
        gray: Graustufenbild als np.ndarray.

    Returns:
        Mittlere Helligkeit als float.
    """
    return float(np.mean(gray))


def calculate_contrast_score(gray: np.ndarray) -> float:
    """Berechnet den Kontrast als Standardabweichung der Pixelwerte (0–128+).

    Args:
        gray: Graustufenbild als np.ndarray.

    Returns:
        Standardabweichung der Pixelwerte als float.
    """
    return float(np.std(gray))


def normalize_to_score(value: float, min_val: float, max_val: float) -> float:
    """Normalisiert einen Wert auf den Bereich 0–100.

    Args:
        value:   Eingabewert.
        min_val: Untere Grenze des erwarteten Bereichs.
        max_val: Obere Grenze des erwarteten Bereichs.

    Returns:
        Normalisierter Score zwischen 0.0 und 100.0.
    """
    if max_val == min_val:
        return 0.0
    score = (value - min_val) / (max_val - min_val) * 100.0
    return float(max(0.0, min(100.0, score)))
