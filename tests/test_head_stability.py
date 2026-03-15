"""pytest-Tests für das head_stability Modul (v0.2 Phase 2)."""

import math
import pytest

from app.cv.head_stability import (
    NOSE_TIP_INDEX,
    DEFAULT_MAX_MOTION_STD,
    StabilityMetrics,
    extract_reference_point_sequence,
    calculate_position_std,
    calculate_head_stability_score,
    label_head_stability,
    summarize_head_stability,
)


# ──────────────────────────────────────────────────────────────
# Hilfsfunktionen für Testdaten
# ──────────────────────────────────────────────────────────────

def _make_landmarks(x: float, y: float, n_landmarks: int = 10) -> list:
    """Erstellt eine minimale Landmark-Liste mit einer fixen Position."""
    lm = [(0.0, 0.0, 0.0)] * n_landmarks
    # Index 0 und 1 mit realen Werten belegen
    lm[0] = (0.5, 0.5, 0.0)
    lm[1] = (x, y, 0.0)   # Nasenspitze (Index 1)
    return lm


def _stable_sequence(n: int = 20) -> list[list]:
    """Erzeugt eine stabile Landmark-Sequenz (Nasenspitze immer an gleicher Position)."""
    return [_make_landmarks(0.50, 0.45) for _ in range(n)]


def _unstable_sequence(amplitude: float = 0.08, n: int = 20) -> list[list]:
    """Erzeugt eine instabile Sequenz mit abwechselnd versetzten Positionen."""
    frames = []
    for i in range(n):
        offset = amplitude if i % 2 == 0 else -amplitude
        frames.append(_make_landmarks(0.50 + offset, 0.45 + offset))
    return frames


def _jitter_sequence(std: float = 0.02, n: int = 30) -> list[list]:
    """Erzeugt eine Sequenz mit normalverteiltem Rauschen."""
    import numpy as np
    rng = np.random.default_rng(42)
    xs = rng.normal(0.5, std, n)
    ys = rng.normal(0.45, std, n)
    return [_make_landmarks(float(x), float(y)) for x, y in zip(xs, ys)]


# ──────────────────────────────────────────────────────────────
# extract_reference_point_sequence
# ──────────────────────────────────────────────────────────────

class TestExtractReferencePointSequence:

    def test_returns_list(self):
        seq = _stable_sequence(5)
        result = extract_reference_point_sequence(seq)
        assert isinstance(result, list)

    def test_correct_length(self):
        seq = _stable_sequence(10)
        result = extract_reference_point_sequence(seq)
        assert len(result) == 10

    def test_empty_input_returns_empty(self):
        result = extract_reference_point_sequence([])
        assert result == []

    def test_skips_empty_frames(self):
        seq = _stable_sequence(5)
        seq_with_empty = seq[:2] + [[]] + seq[2:]   # leerer Frame eingebaut
        result = extract_reference_point_sequence(seq_with_empty)
        assert len(result) == 5   # leerer Frame übersprungen

    def test_values_are_tuples_of_two_floats(self):
        seq = _stable_sequence(3)
        result = extract_reference_point_sequence(seq)
        for pos in result:
            assert len(pos) == 2
            assert isinstance(pos[0], float)
            assert isinstance(pos[1], float)

    def test_correct_position_extracted(self):
        lm = _make_landmarks(0.42, 0.38)
        result = extract_reference_point_sequence([lm], landmark_index=NOSE_TIP_INDEX)
        assert result[0] == (0.42, 0.38)

    def test_custom_landmark_index(self):
        lm = _make_landmarks(0.5, 0.5)
        # Index 0 hat (0.5, 0.5) aus _make_landmarks
        result = extract_reference_point_sequence([lm], landmark_index=0)
        assert result[0][0] == pytest.approx(0.5)

    def test_out_of_range_index_skipped(self):
        lm = [(0.5, 0.5, 0.0)]  # Nur 1 Landmark
        result = extract_reference_point_sequence([lm], landmark_index=100)
        assert result == []


# ──────────────────────────────────────────────────────────────
# calculate_position_std
# ──────────────────────────────────────────────────────────────

class TestCalculatePositionStd:

    def test_returns_stability_metrics(self):
        positions = [(0.5, 0.4), (0.51, 0.41)]
        result = calculate_position_std(positions)
        assert isinstance(result, StabilityMetrics)

    def test_stable_sequence_near_zero_std(self):
        positions = [(0.5, 0.4)] * 20
        result = calculate_position_std(positions)
        assert result.std_x == pytest.approx(0.0, abs=1e-9)
        assert result.std_y == pytest.approx(0.0, abs=1e-9)
        assert result.combined_motion_std == pytest.approx(0.0, abs=1e-9)

    def test_empty_positions_not_reliable(self):
        result = calculate_position_std([])
        assert result.is_reliable is False
        assert result.frame_count == 0
        assert result.combined_motion_std == 0.0

    def test_single_position_not_reliable(self):
        result = calculate_position_std([(0.5, 0.4)])
        assert result.is_reliable is False
        assert result.frame_count == 1

    def test_two_positions_is_reliable(self):
        result = calculate_position_std([(0.4, 0.3), (0.6, 0.5)])
        assert result.is_reliable is True

    def test_frame_count_correct(self):
        positions = [(0.5, 0.4)] * 15
        result = calculate_position_std(positions)
        assert result.frame_count == 15

    def test_combined_std_is_euclidean(self):
        positions = [(0.0, 0.0), (1.0, 0.0)]  # std_x ≈ 0.707, std_y = 0
        result = calculate_position_std(positions)
        expected_combined = math.sqrt(result.std_x ** 2 + result.std_y ** 2)
        assert result.combined_motion_std == pytest.approx(expected_combined, rel=1e-5)

    def test_std_non_negative(self):
        positions = [(0.1 * i, 0.2 * i) for i in range(10)]
        result = calculate_position_std(positions)
        assert result.std_x >= 0
        assert result.std_y >= 0
        assert result.combined_motion_std >= 0

    def test_unstable_has_larger_std_than_stable(self):
        stable = [(0.5, 0.45)] * 20
        unstable = [(0.3 + 0.1 * (i % 3), 0.4 + 0.05 * (i % 4)) for i in range(20)]
        r_stable = calculate_position_std(stable)
        r_unstable = calculate_position_std(unstable)
        assert r_unstable.combined_motion_std > r_stable.combined_motion_std


# ──────────────────────────────────────────────────────────────
# calculate_head_stability_score
# ──────────────────────────────────────────────────────────────

class TestCalculateHeadStabilityScore:

    def test_zero_std_gives_100(self):
        score = calculate_head_stability_score(0.0)
        assert score == pytest.approx(100.0)

    def test_max_std_gives_zero(self):
        score = calculate_head_stability_score(DEFAULT_MAX_MOTION_STD)
        assert score == pytest.approx(0.0)

    def test_half_std_gives_50(self):
        score = calculate_head_stability_score(DEFAULT_MAX_MOTION_STD / 2)
        assert score == pytest.approx(50.0, abs=0.1)

    def test_over_max_clipped_to_zero(self):
        score = calculate_head_stability_score(DEFAULT_MAX_MOTION_STD * 2)
        assert score == 0.0

    def test_score_always_in_range(self):
        for std in [0.0, 0.01, 0.025, 0.05, 0.1, 0.5]:
            score = calculate_head_stability_score(std)
            assert 0.0 <= score <= 100.0

    def test_custom_max_std(self):
        score = calculate_head_stability_score(0.10, max_motion_std=0.20)
        assert score == pytest.approx(50.0, abs=0.1)

    def test_negative_std_clipped_to_100(self):
        # Negativer Wert (sollte in Praxis nicht vorkommen) → Score = 100
        score = calculate_head_stability_score(-0.01)
        assert score == 100.0

    def test_monotonically_decreasing(self):
        stds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        scores = [calculate_head_stability_score(s) for s in stds]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ──────────────────────────────────────────────────────────────
# label_head_stability
# ──────────────────────────────────────────────────────────────

class TestLabelHeadStability:

    _VALID_LABELS = {"Sehr stabil", "Stabil", "Mäßig stabil", "Instabil", "Sehr instabil"}

    def test_high_score_label(self):
        assert label_head_stability(90.0) == "Sehr stabil"

    def test_medium_high_score_label(self):
        assert label_head_stability(70.0) == "Stabil"

    def test_medium_score_label(self):
        assert label_head_stability(50.0) == "Mäßig stabil"

    def test_low_score_label(self):
        assert label_head_stability(30.0) == "Instabil"

    def test_very_low_score_label(self):
        assert label_head_stability(10.0) == "Sehr instabil"

    def test_zero_score_label(self):
        assert label_head_stability(0.0) == "Sehr instabil"

    def test_100_score_label(self):
        assert label_head_stability(100.0) == "Sehr stabil"

    @pytest.mark.parametrize("score", [0, 10, 25, 40, 55, 70, 85, 100])
    def test_all_scores_return_valid_label(self, score):
        label = label_head_stability(float(score))
        assert label in self._VALID_LABELS

    def test_labels_are_german(self):
        for score in range(0, 101, 10):
            label = label_head_stability(float(score))
            assert isinstance(label, str)
            assert len(label) > 0


# ──────────────────────────────────────────────────────────────
# summarize_head_stability (Integrations-Tests)
# ──────────────────────────────────────────────────────────────

class TestSummarizeHeadStability:

    def test_returns_dict(self):
        result = summarize_head_stability(_stable_sequence(10))
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = summarize_head_stability(_stable_sequence(5))
        required = {
            "head_stability_score", "label", "std_x", "std_y",
            "combined_motion_std", "frame_count", "is_reliable",
            "landmark_index", "warnung",
        }
        assert required.issubset(result.keys())

    def test_stable_sequence_high_score(self):
        result = summarize_head_stability(_stable_sequence(20))
        assert result["head_stability_score"] >= 80.0

    def test_unstable_sequence_low_score(self):
        result = summarize_head_stability(_unstable_sequence(amplitude=0.06, n=20))
        assert result["head_stability_score"] < 50.0

    def test_stable_has_higher_score_than_unstable(self):
        r_stable = summarize_head_stability(_stable_sequence(20))
        r_unstable = summarize_head_stability(_unstable_sequence(amplitude=0.07, n=20))
        assert r_stable["head_stability_score"] > r_unstable["head_stability_score"]

    def test_jitter_sequence_score_in_range(self):
        result = summarize_head_stability(_jitter_sequence(std=0.01, n=30))
        assert 0.0 <= result["head_stability_score"] <= 100.0

    def test_empty_input_defensive(self):
        result = summarize_head_stability([])
        assert result["head_stability_score"] == pytest.approx(100.0)
        assert result["is_reliable"] is False
        assert result["frame_count"] == 0
        assert result["warnung"] is not None

    def test_single_frame_not_reliable(self):
        result = summarize_head_stability(_stable_sequence(1))
        assert result["is_reliable"] is False
        assert result["warnung"] is not None

    def test_two_frames_reliable(self):
        result = summarize_head_stability(_stable_sequence(2))
        assert result["is_reliable"] is True

    def test_score_always_in_0_100(self):
        for seq in [
            _stable_sequence(20),
            _unstable_sequence(amplitude=0.10, n=20),
            _jitter_sequence(std=0.03, n=15),
            [],
            _stable_sequence(1),
        ]:
            result = summarize_head_stability(seq)
            assert 0.0 <= result["head_stability_score"] <= 100.0

    def test_warnung_none_for_reliable_data(self):
        result = summarize_head_stability(_stable_sequence(20))
        assert result["warnung"] is None

    def test_warnung_string_for_empty(self):
        result = summarize_head_stability([])
        assert isinstance(result["warnung"], str)
        assert len(result["warnung"]) > 0

    def test_landmark_index_recorded(self):
        result = summarize_head_stability(_stable_sequence(5), landmark_index=4)
        assert result["landmark_index"] == 4

    def test_default_landmark_is_nose_tip(self):
        result = summarize_head_stability(_stable_sequence(5))
        assert result["landmark_index"] == NOSE_TIP_INDEX

    def test_frame_count_matches_input(self):
        result = summarize_head_stability(_stable_sequence(15))
        assert result["frame_count"] == 15
