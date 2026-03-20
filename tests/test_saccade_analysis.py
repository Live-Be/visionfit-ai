"""Tests für app.cv.saccade_analysis und app.scoring.rules.score_saccade_test.

Testet:
    - extract_iris_x_sequence  (Iris-Extraktion aus synthetischen Frames)
    - analyze_saccade_test      (End-to-End-Analyse mit Stub-Daten)
    - score_saccade_test        (Scoring-Logik)

Kein Streamlit-Import, keine reale Kamera erforderlich.
Synthetische schwarze Frames (kein Gesicht) werden verwendet um die
Fehlerpfade und Grundstruktur der Ausgabe zu testen.
Die eigentliche Iris-Tracking-Logik kann nur mit echten Gesichtsbildern
vollständig getestet werden – daher liegt der Schwerpunkt auf:
    1. Korrekte Ausgabestruktur (Keys, Typen, Wertebereiche)
    2. Robustheit bei fehlenden / leeren Daten
    3. Scoring-Formel (vollständig ohne CV möglich)
    4. Hilfsfunktionen (EMA-Glättung, Korrektursakkaden-Zählung)
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.cv.saccade_analysis import (
    MIN_IRIS_DETECTION_RATE,
    _count_correction_saccades,
    _ema_smooth,
    _extract_iris_x,
    _find_saccade_onset,
    _interpret_results,
    analyze_saccade_test,
    extract_iris_x_sequence,
)
from app.scoring.rules import score_saccade_test


# ──────────────────────────────────────────────────────────────────────────────
# Test-Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _black_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_timed_frames(n: int = 10, fps: float = 30.0) -> list[tuple[np.ndarray, float]]:
    """Erstellt synthetische (frame, timestamp) Paare."""
    t0 = 1000.0  # Beliebiger Startzeitpunkt
    dt = 1.0 / fps
    return [(_black_frame(), t0 + i * dt) for i in range(n)]


def _make_stimulus_events(n: int = 5, start: float = 1000.0) -> list[dict]:
    """Erstellt eine einfache Test-Stimulus-Sequenz."""
    events = []
    t = start
    direction = "left"
    for _ in range(n):
        events.append({"time": t, "direction": direction})
        t += 2.0
        direction = "right" if direction == "left" else "left"
    return events


# ──────────────────────────────────────────────────────────────────────────────
# 1. _extract_iris_x
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractIrisX:
    def test_none_for_empty_landmarks(self):
        assert _extract_iris_x([]) is None

    def test_none_for_short_landmark_list(self):
        # Weniger als 474 Landmarks → keine Iris
        short = [(0.5, 0.5, 0.0)] * 470
        assert _extract_iris_x(short) is None

    def test_returns_float_for_valid_landmarks(self):
        # 478 Landmarks (refine_landmarks=True)
        lm = [(0.5, 0.5, 0.0)] * 478
        result = _extract_iris_x(lm)
        assert isinstance(result, float)

    def test_value_in_range(self):
        lm = [(0.3, 0.5, 0.0)] * 478
        result = _extract_iris_x(lm)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_averages_both_irises(self):
        lm = [(0.0, 0.5, 0.0)] * 478
        # Left iris center (468) = 0.2, right iris center (473) = 0.6
        lm[468] = (0.2, 0.5, 0.0)
        lm[473] = (0.6, 0.5, 0.0)
        result = _extract_iris_x(lm)
        assert result == pytest.approx(0.4)

    def test_none_for_none_input(self):
        assert _extract_iris_x(None) is None  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# 2. _ema_smooth
# ──────────────────────────────────────────────────────────────────────────────

class TestEmaSmooth:
    def test_empty_input(self):
        assert _ema_smooth([]) == []

    def test_same_length_as_input(self):
        signal = [0.1, 0.2, None, 0.4, 0.5]
        result = _ema_smooth(signal)
        assert len(result) == len(signal)

    def test_none_values_propagated(self):
        # Erster Wert None → bleibt None
        result = _ema_smooth([None, None, 0.5])
        assert result[0] is None
        assert result[1] is None
        assert result[2] is not None

    def test_no_nones_in_output_after_first_value(self):
        result = _ema_smooth([0.5, None, None, 0.3])
        # Nach dem ersten Wert keine None mehr
        assert all(v is not None for v in result[1:])

    def test_smoothing_reduces_variance(self):
        # Starkes Rauschen → nach EMA weniger Varianz
        noisy = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        smoothed = _ema_smooth(noisy, alpha=0.3)
        noisy_std = float(np.std([v for v in noisy if v is not None]))
        smooth_std = float(np.std([v for v in smoothed if v is not None]))
        assert smooth_std < noisy_std

    def test_constant_signal_unchanged(self):
        signal = [0.5] * 5
        result = _ema_smooth(signal, alpha=0.3)
        for v in result:
            assert v == pytest.approx(0.5, abs=1e-6)

    def test_alpha_1_equals_input(self):
        signal = [0.1, 0.5, 0.3]
        result = _ema_smooth(signal, alpha=1.0)
        for orig, res in zip(signal, result):
            assert res == pytest.approx(orig, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 3. _find_saccade_onset
# ──────────────────────────────────────────────────────────────────────────────

class TestFindSaccadeOnset:
    def test_returns_none_for_no_movement(self):
        # Kein Signal → kein Onset
        gaze = [0.5, 0.5, 0.5, 0.5, 0.5]
        times = [0.0, 0.05, 0.1, 0.15, 0.2]
        result = _find_saccade_onset(gaze, times, event_time=0.0)
        assert result is None

    def test_detects_fast_movement(self):
        # Starke Bewegung bei t=0.1s
        gaze = [0.5, 0.5, 0.5, 0.8, 0.8]
        times = [0.0, 0.05, 0.1, 0.15, 0.2]
        result = _find_saccade_onset(gaze, times, event_time=0.0)
        assert result is not None
        assert result == pytest.approx(0.15)

    def test_ignores_movement_strictly_before_event(self):
        # Alle Frames vor event_time → kein Suchfenster → None
        gaze = [0.5, 0.9, 0.8, 0.7, 0.6]
        times = [0.0, 0.05, 0.1, 0.12, 0.14]
        result = _find_saccade_onset(gaze, times, event_time=0.15)
        assert result is None

    def test_stable_pre_event_means_movement_detected_post_event(self):
        # Stabiles Auge vor dem Event, dann schnelle Bewegung nach Event
        # → soll als Sakkade erkannt werden
        gaze = [0.3, 0.3, 0.3, 0.7, 0.7]
        times = [0.0, 0.05, 0.1, 0.2, 0.25]
        result = _find_saccade_onset(gaze, times, event_time=0.15)
        assert result == pytest.approx(0.2)

    def test_ignores_movement_after_window(self):
        # Bewegung außerhalb des Suchfensters
        gaze = [0.5, 0.5, 0.5, 0.5, 0.9]
        times = [0.0, 0.1, 0.2, 0.3, 1.5]  # 1.5s > search_window_s=0.6
        result = _find_saccade_onset(gaze, times, event_time=0.0, search_window_s=0.6)
        assert result is None

    def test_handles_none_in_gaze(self):
        # None-Werte sollen überspringen, nicht crashen
        gaze = [None, 0.5, None, 0.9, 0.9]
        times = [0.0, 0.05, 0.1, 0.15, 0.2]
        result = _find_saccade_onset(gaze, times, event_time=0.0)
        # Kann None oder einen Wert zurückgeben – soll jedenfalls nicht crashen
        assert result is None or isinstance(result, float)


# ──────────────────────────────────────────────────────────────────────────────
# 4. _count_correction_saccades
# ──────────────────────────────────────────────────────────────────────────────

class TestCountCorrectionSaccades:
    def test_no_corrections_for_stable_signal(self):
        gaze = [0.7, 0.7, 0.7, 0.7]
        times = [0.3, 0.35, 0.4, 0.45]
        result = _count_correction_saccades(gaze, times, 0.25, 0.8)
        assert result == 0

    def test_detects_direction_reversal(self):
        # Starker Richtungswechsel → 1 Korrektur
        gaze = [0.5, 0.6, 0.7, 0.6, 0.5]
        times = [0.3, 0.35, 0.4, 0.45, 0.5]
        result = _count_correction_saccades(gaze, times, 0.25, 0.8, min_amplitude=0.05)
        assert result >= 1

    def test_too_few_frames_returns_zero(self):
        gaze = [0.5]
        times = [0.3]
        result = _count_correction_saccades(gaze, times, 0.25, 0.8)
        assert result == 0

    def test_handles_none_values(self):
        gaze = [None, 0.5, None, 0.6]
        times = [0.3, 0.35, 0.4, 0.45]
        # Soll nicht crashen
        result = _count_correction_saccades(gaze, times, 0.25, 0.8)
        assert isinstance(result, int)
        assert result >= 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. extract_iris_x_sequence
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractIrisXSequence:
    def test_empty_input(self):
        result = extract_iris_x_sequence([])
        assert result["iris_x_raw"] == []
        assert result["iris_detection_rate"] == 0.0

    def test_returns_required_keys(self):
        result = extract_iris_x_sequence([_black_frame()])
        assert {"iris_x_raw", "iris_x_smooth", "face_detection_rate",
                "iris_detection_rate", "frames_landmarks"} <= result.keys()

    def test_black_frames_no_iris(self):
        frames = [_black_frame()] * 5
        result = extract_iris_x_sequence(frames)
        assert result["iris_detection_rate"] == 0.0

    def test_length_matches_input(self):
        frames = [_black_frame()] * 7
        result = extract_iris_x_sequence(frames)
        assert len(result["iris_x_raw"]) == 7
        assert len(result["iris_x_smooth"]) == 7

    def test_detection_rate_in_range(self):
        frames = [_black_frame()] * 5
        result = extract_iris_x_sequence(frames)
        assert 0.0 <= result["iris_detection_rate"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 6. analyze_saccade_test (Fehlerpfade + Ausgabestruktur)
# ──────────────────────────────────────────────────────────────────────────────

_EXPECTED_KEYS = {
    "latency_ms_mean", "latency_ms_std", "accuracy_score",
    "correction_saccades_count", "symmetry_score", "quality_score",
    "interpretation_text", "raw_event_count", "analyzed_event_count",
    "face_detection_rate", "iris_detection_rate", "head_movement_warning",
    "is_reliable", "warnung",
}


class TestAnalyzeSaccadeTest:
    def test_empty_frames_returns_error(self):
        result = analyze_saccade_test([], _make_stimulus_events())
        assert result["warnung"] is not None
        assert result["is_reliable"] is False

    def test_empty_events_returns_error(self):
        result = analyze_saccade_test(_make_timed_frames(), [])
        assert result["warnung"] is not None
        assert result["is_reliable"] is False

    def test_returns_all_required_keys(self):
        result = analyze_saccade_test(_make_timed_frames(), _make_stimulus_events())
        assert _EXPECTED_KEYS <= result.keys()

    def test_black_frames_low_iris_detection(self):
        # Schwarze Frames → kein Gesicht → iris_detection_rate = 0 → warnung
        result = analyze_saccade_test(_make_timed_frames(30), _make_stimulus_events())
        assert result["iris_detection_rate"] < MIN_IRIS_DETECTION_RATE
        assert result["warnung"] is not None

    def test_accuracy_score_in_range(self):
        result = analyze_saccade_test(_make_timed_frames(), _make_stimulus_events())
        assert 0.0 <= result["accuracy_score"] <= 100.0

    def test_quality_score_in_range(self):
        result = analyze_saccade_test(_make_timed_frames(), _make_stimulus_events())
        assert 0.0 <= result["quality_score"] <= 100.0

    def test_correction_count_non_negative(self):
        result = analyze_saccade_test(_make_timed_frames(), _make_stimulus_events())
        assert result["correction_saccades_count"] >= 0

    def test_is_reliable_false_for_black_frames(self):
        result = analyze_saccade_test(_make_timed_frames(30), _make_stimulus_events())
        assert result["is_reliable"] is False

    def test_head_movement_warning_is_bool(self):
        result = analyze_saccade_test(_make_timed_frames(10), _make_stimulus_events())
        assert isinstance(result["head_movement_warning"], bool)

    def test_raw_event_count_correct(self):
        events = _make_stimulus_events(8)
        result = analyze_saccade_test(_make_timed_frames(30), events)
        assert result["raw_event_count"] == 8

    def test_interpretation_text_is_string(self):
        result = analyze_saccade_test(_make_timed_frames(), _make_stimulus_events())
        assert isinstance(result["interpretation_text"], str)

    def test_no_crash_on_minimal_input(self):
        """Robustheit: minimale Eingabe darf nicht crashen."""
        timed = [(_black_frame(), 1000.0)]
        events = [{"time": 1000.0, "direction": "left"}]
        result = analyze_saccade_test(timed, events)
        assert isinstance(result, dict)


# ──────────────────────────────────────────────────────────────────────────────
# 7. score_saccade_test
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreSaccadeTest:
    def test_returns_score_result_keys(self):
        result = score_saccade_test(
            accuracy_score=80.0,
            latency_ms_mean=200.0,
            correction_saccades_count=2,
            quality_score=90.0,
            is_reliable=True,
        )
        assert {"score", "label", "details"} <= result.keys()

    def test_score_in_range(self):
        result = score_saccade_test(
            accuracy_score=80.0,
            latency_ms_mean=200.0,
            correction_saccades_count=2,
            quality_score=90.0,
            is_reliable=True,
        )
        assert 0.0 <= result["score"] <= 100.0

    def test_score_is_float(self):
        result = score_saccade_test(
            accuracy_score=70.0, latency_ms_mean=220.0,
            correction_saccades_count=1, quality_score=85.0, is_reliable=True,
        )
        assert isinstance(result["score"], float)

    def test_label_is_string(self):
        result = score_saccade_test(
            accuracy_score=70.0, latency_ms_mean=220.0,
            correction_saccades_count=1, quality_score=85.0, is_reliable=True,
        )
        assert isinstance(result["label"], str)

    def test_perfect_inputs_high_score(self):
        result = score_saccade_test(
            accuracy_score=100.0,
            latency_ms_mean=150.0,
            correction_saccades_count=0,
            quality_score=100.0,
            is_reliable=True,
        )
        assert result["score"] >= 80.0

    def test_poor_inputs_low_score(self):
        result = score_saccade_test(
            accuracy_score=0.0,
            latency_ms_mean=500.0,
            correction_saccades_count=10,
            quality_score=20.0,
            is_reliable=True,
        )
        assert result["score"] <= 40.0

    def test_none_latency_accepted(self):
        result = score_saccade_test(
            accuracy_score=70.0, latency_ms_mean=None,
            correction_saccades_count=0, quality_score=80.0, is_reliable=True,
        )
        assert 0.0 <= result["score"] <= 100.0

    def test_unreliable_reduces_score(self):
        reliable = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=200.0,
            correction_saccades_count=2, quality_score=90.0, is_reliable=True,
        )
        unreliable = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=200.0,
            correction_saccades_count=2, quality_score=90.0, is_reliable=False,
        )
        assert unreliable["score"] <= reliable["score"]

    def test_details_contains_expected_keys(self):
        result = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=200.0,
            correction_saccades_count=2, quality_score=90.0, is_reliable=True,
        )
        assert "accuracy_score" in result["details"]
        assert "correction_saccades" in result["details"]
        assert "gewichtung" in result["details"]

    def test_more_corrections_lower_score(self):
        few = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=200.0,
            correction_saccades_count=1, quality_score=90.0, is_reliable=True,
        )
        many = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=200.0,
            correction_saccades_count=8, quality_score=90.0, is_reliable=True,
        )
        assert many["score"] < few["score"]

    def test_lower_latency_higher_score(self):
        fast = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=150.0,
            correction_saccades_count=1, quality_score=90.0, is_reliable=True,
        )
        slow = score_saccade_test(
            accuracy_score=80.0, latency_ms_mean=450.0,
            correction_saccades_count=1, quality_score=90.0, is_reliable=True,
        )
        assert fast["score"] > slow["score"]

    @pytest.mark.parametrize("accuracy,latency,corrections", [
        (0.0, 0.0, 0),
        (100.0, 600.0, 15),
        (50.0, None, 5),
        (100.0, 150.0, 0),
    ])
    def test_extreme_values_stay_in_range(self, accuracy, latency, corrections):
        result = score_saccade_test(
            accuracy_score=accuracy,
            latency_ms_mean=latency,
            correction_saccades_count=corrections,
            quality_score=100.0,
            is_reliable=True,
        )
        assert 0.0 <= result["score"] <= 100.0


# ──────────────────────────────────────────────────────────────────────────────
# 8. _interpret_results
# ──────────────────────────────────────────────────────────────────────────────

class TestInterpretResults:
    def test_too_few_events(self):
        result = _interpret_results(200.0, 80.0, 1, 85.0, n_analyzed=2)
        assert "wenige" in result.lower() or "zu wenige" in result.lower()

    def test_returns_string(self):
        result = _interpret_results(200.0, 80.0, 1, 85.0, n_analyzed=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fast_latency_mentioned(self):
        result = _interpret_results(150.0, 80.0, 0, 90.0, n_analyzed=8)
        assert "schnell" in result.lower()

    def test_slow_latency_mentioned(self):
        result = _interpret_results(400.0, 80.0, 0, 90.0, n_analyzed=8)
        assert "verzögert" in result.lower()

    def test_none_latency_handled(self):
        result = _interpret_results(None, 80.0, 0, 90.0, n_analyzed=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_many_corrections_mentioned(self):
        result = _interpret_results(200.0, 80.0, 8, 90.0, n_analyzed=8)
        assert "viele" in result.lower()

    def test_symmetry_none_no_crash(self):
        result = _interpret_results(200.0, 80.0, 1, None, n_analyzed=5)
        assert isinstance(result, str)
