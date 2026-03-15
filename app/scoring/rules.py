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


def score_fixation_with_stability(
    brightness: float,
    contrast: float,
    head_stability_score: float | None = None,
) -> ScoreResult:
    """Bewertet den Fixationstest mit optionalem Kopfstabilitäts-Einfluss.

    Wenn head_stability_score übergeben wird, fließt er mit 30 % Gewicht
    in den Gesamtscore ein (Bild-Score 70 %, Stabilitäts-Score 30 %).
    Ohne Stabilitätsscore entspricht das Verhalten score_fixation_test().

    HINWEIS: Setzt voraus, dass head_stability_score aus mindestens 2 Frames
    berechnet wurde. Bei Einzelbild-Aufnahmen ist dieser Wert rein formal.

    Args:
        brightness:           Mittlere Helligkeit (0–255).
        contrast:             Standardabweichung Pixelwerte.
        head_stability_score: Kopfstabilitätsscore (0–100) oder None.

    Returns:
        ScoreResult mit score (0–100), label und details.
    """
    base = score_fixation_test(brightness=brightness, contrast=contrast)

    if head_stability_score is None:
        return base

    head_stability_score = max(0.0, min(100.0, float(head_stability_score)))

    # Gewichtete Kombination: 70 % Bild-Score + 30 % Stabilitätsscore
    combined = round(base["score"] * 0.70 + head_stability_score * 0.30, 1)

    return ScoreResult(
        score=combined,
        label=_label_for_score(combined),
        details={
            **base["details"],
            "head_stability_score": round(head_stability_score, 1),
            "gewichtung": "70% Bild + 30% Stabilität",
        },
    )


def blink_rate_adjustment(blink_rate_per_min: float | None) -> float:
    """Berechnet einen Anpassungsfaktor für den Fixationsscore basierend auf Blinkrate.

    Vorbereitung für v0.3 (Multi-Frame): Gibt einen Faktor zurück mit dem der
    Fixationsscore optional korrigiert werden kann. Ohne Blinkrate (None) → 0.0
    (kein Einfluss). Bei normaler Blinkrate → 0.0. Bei Extremwerten → leichte
    negative Korrektur.

    HINWEIS: Nur belastbar mit Multi-Frame-Daten (is_reliable=True aus
    summarize_eye_metrics). Bei Einzelbild-Ergebnissen nicht verwenden.

    Heuristik (nicht klinisch validiert):
    - Normale Blinkrate: 10–25/min → Anpassung: 0.0
    - Sehr niedrig (< 5/min) oder sehr hoch (> 40/min) → Anpassung: -5.0
    - Dazwischen: lineare Interpolation

    Args:
        blink_rate_per_min:  Blinkrate in Blinks/Minute oder None.

    Returns:
        Score-Anpassung als float (typisch 0.0 bis -5.0).
    """
    if blink_rate_per_min is None:
        return 0.0

    rate = float(blink_rate_per_min)

    # Normalbereich: kein Einfluss
    if 10.0 <= rate <= 25.0:
        return 0.0

    # Sehr niedriger oder sehr hoher Bereich: leichte Penalty
    if rate < 5.0 or rate > 40.0:
        return -5.0

    # Übergangsbereich: sanfte lineare Interpolation
    if rate < 10.0:
        return round(-5.0 * (1.0 - (rate - 5.0) / 5.0), 1)

    # rate > 25.0 und <= 40.0
    return round(-5.0 * ((rate - 25.0) / 15.0), 1)


def score_fixation_combined(
    brightness: float,
    contrast: float,
    head_stability_score: float,
    blink_rate: float | None = None,
) -> ScoreResult:
    """Bewertet den Fixationstest mit Stabilität und Blinkmuster (v0.3 Multi-Frame).

    Kombiniert drei Komponenten zu einem Gesamtscore:
        - Bildqualität:    50 % (Helligkeit + Kontrast)
        - Kopfstabilität:  30 % (Nasenspitzen-Tracking über Frames)
        - Blinkmuster:     20 % (normale Blinkrate = voller Beitrag)

    Blink-Einfluss: blink_rate_adjustment() liefert einen Faktor in [-5.0, 0.0].
    Skalierung auf Blink-Score: blink_score = 100 + adj × 4 → Bereich [80, 100].
    Damit reduziert eine abnormale Blinkrate den Gesamtscore maximal um ~4 Punkte
    (Stressindikator, „leicht reduzieren").

    HINWEIS: Heuristische Analyse, nicht klinisch validiert.
    Nur belastbar mit Multi-Frame-Daten (is_reliable=True aus
    summarize_eye_metrics und summarize_head_stability).

    Args:
        brightness:           Mittlere Helligkeit des mittleren Frames (0–255).
        contrast:             Standardabweichung der Pixelwerte.
        head_stability_score: Kopfstabilitätsscore (0–100).
        blink_rate:           Blinkrate in Blinks/Minute oder None (kein Einfluss).

    Returns:
        ScoreResult mit score (0–100), label und details.
    """
    # Bildqualitäts-Score (50 %)
    image_result = score_fixation_test(brightness=brightness, contrast=contrast)
    image_score = image_result["score"]

    # Stabilitäts-Score (30 %, geclippt auf [0, 100])
    stability = max(0.0, min(100.0, float(head_stability_score)))

    # Blink-Score (20 %): adj ∈ [-5.0, 0.0] → blink_score ∈ [80.0, 100.0]
    blink_adj = blink_rate_adjustment(blink_rate)
    blink_score = max(0.0, min(100.0, 100.0 + blink_adj * 4.0))

    # Gewichtete Kombination: 50% Bild + 30% Stabilität + 20% Blinkmuster
    combined = round(
        image_score * 0.50 + stability * 0.30 + blink_score * 0.20,
        1,
    )
    combined = max(0.0, min(100.0, combined))

    return ScoreResult(
        score=combined,
        label=_label_for_score(combined),
        details={
            **image_result["details"],
            "head_stability_score": round(stability, 1),
            "blink_rate_pro_min": round(blink_rate, 1) if blink_rate is not None else None,
            "blink_score": round(blink_score, 1),
            "blink_anpassung": round(blink_adj, 1),
            "gewichtung": "50% Bildqualität + 30% Stabilität + 20% Blinkmuster",
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
