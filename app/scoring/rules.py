"""Scoring-Regeln für Fixations- und Lese-Komfort-Test."""

from __future__ import annotations
from typing import TypedDict

# Spezielles Label für ungültige Tests (z.B. kein Gesicht erkannt)
LABEL_INVALID = "Test ungültig – kein Gesicht erkannt"


class ScoreResult(TypedDict):
    score: float
    label: str
    details: dict


def score_fixation_no_face() -> ScoreResult:
    """Gibt ein Ungültig-Ergebnis zurück wenn kein Gesicht erkannt wurde.

    Returns:
        ScoreResult mit score=0 und entsprechendem Label.
    """
    return ScoreResult(
        score=0.0,
        label=LABEL_INVALID,
        details={"grund": "Kein Gesicht im Bild erkannt."},
    )


def _label_for_score(score: float) -> str:
    """Gibt ein deutsches Label für einen Score-Wert zurück."""
    if score >= 80:
        return "Sehr gut"
    if score >= 60:
        return "Gut"
    if score >= 40:
        return "Mittel"
    if score >= 20:
        return "Schwach"
    return "Ungenügend"


def score_fixation_test(
    brightness: float,
    contrast: float,
) -> ScoreResult:
    """Bewertet den Fixationstest anhand von Helligkeits- und Kontrastwerten.

    Heuristik:
    - Optimale Helligkeit: 80–180 (Strafe bei Über-/Unterbelichtung)
    - Kontrast: je höher, desto besser (max. ~60 als Referenz)

    Args:
        brightness: Mittlere Helligkeit des aufgenommenen Bildes (0–255).
        contrast:   Standardabweichung der Pixelwerte des aufgenommenen Bildes.

    Returns:
        ScoreResult mit score (0–100), label und details.
    """
    # Helligkeitspenalty: Abstand vom Optimalbereich [80, 180]
    optimal_low, optimal_high = 80.0, 180.0
    if brightness < optimal_low:
        brightness_penalty = (optimal_low - brightness) / optimal_low * 50
    elif brightness > optimal_high:
        brightness_penalty = (brightness - optimal_high) / (255 - optimal_high) * 50
    else:
        brightness_penalty = 0.0

    # Kontrast-Score (normiert auf 0–50, Referenzmax = 60)
    contrast_score = min(contrast / 60.0, 1.0) * 50.0

    raw_score = max(0.0, 50.0 - brightness_penalty) + contrast_score
    score = round(min(100.0, max(0.0, raw_score)), 1)

    return ScoreResult(
        score=score,
        label=_label_for_score(score),
        details={
            "helligkeit": round(brightness, 1),
            "kontrast": round(contrast, 1),
            "helligkeitspenalty": round(brightness_penalty, 1),
            "kontrast_score": round(contrast_score, 1),
        },
    )


def score_reading_test(
    anstrengung: int,
    unschaerfe: int,
    komfort: int,
) -> ScoreResult:
    """Bewertet den Lese-Komfort-Test anhand von Slider-Werten.

    Skala der Eingaben: 0–10 (höher = schlechter, außer komfort).

    Heuristik:
    - Anstrengung und Unschärfe werden negativ gewichtet.
    - Komfort wird positiv gewichtet.
    - Gesamtscore = 100 − weighted_penalty + comfort_bonus

    Args:
        anstrengung: Wahrgenommene Anstrengung (0 = keine, 10 = sehr hoch).
        unschaerfe:  Wahrgenommene Unschärfe (0 = keine, 10 = stark).
        komfort:     Wahrgenommener Komfort (0 = sehr unbequem, 10 = sehr bequem).

    Returns:
        ScoreResult mit score (0–100), label und details.
    """
    # Normierung aller Werte auf 0–1
    anstrengung_n = anstrengung / 10.0
    unschaerfe_n = unschaerfe / 10.0
    komfort_n = komfort / 10.0

    # Gewichtete Berechnung
    penalty = (anstrengung_n * 0.4 + unschaerfe_n * 0.4) * 80.0
    bonus = komfort_n * 20.0

    score = round(min(100.0, max(0.0, 100.0 - penalty + bonus - 20.0)), 1)

    return ScoreResult(
        score=score,
        label=_label_for_score(score),
        details={
            "anstrengung": anstrengung,
            "unschaerfe": unschaerfe,
            "komfort": komfort,
            "penalty": round(penalty, 1),
            "bonus": round(bonus, 1),
        },
    )
