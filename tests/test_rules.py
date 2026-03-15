"""pytest-Tests für das Scoring-Regelwerk."""

import pytest
from app.scoring.rules import (
    score_fixation_test,
    score_fixation_no_face,
    score_reading_test,
    LABEL_INVALID,
)


# ──────────────────────────────────────────────
# Fixationstest-Tests
# ──────────────────────────────────────────────

class TestScoreFixationTest:

    def test_score_is_within_range(self):
        result = score_fixation_test(brightness=120.0, contrast=30.0)
        assert 0 <= result["score"] <= 100

    def test_good_conditions_yield_high_score(self):
        result = score_fixation_test(brightness=130.0, contrast=55.0)
        assert result["score"] >= 60

    def test_very_dark_image_yields_lower_score(self):
        result_dark = score_fixation_test(brightness=10.0, contrast=5.0)
        result_good = score_fixation_test(brightness=130.0, contrast=40.0)
        assert result_dark["score"] < result_good["score"]

    def test_very_bright_image_penalty(self):
        result = score_fixation_test(brightness=250.0, contrast=10.0)
        assert result["score"] < 70

    def test_result_has_required_keys(self):
        result = score_fixation_test(brightness=100.0, contrast=25.0)
        assert "score" in result
        assert "label" in result
        assert "details" in result

    def test_label_is_string(self):
        result = score_fixation_test(brightness=100.0, contrast=25.0)
        assert isinstance(result["label"], str)
        assert len(result["label"]) > 0

    @pytest.mark.parametrize("brightness,contrast", [
        (0.0, 0.0),
        (255.0, 0.0),
        (128.0, 128.0),
        (80.0, 60.0),
    ])
    def test_extreme_values_stay_in_range(self, brightness, contrast):
        result = score_fixation_test(brightness=brightness, contrast=contrast)
        assert 0 <= result["score"] <= 100


# ──────────────────────────────────────────────
# Lesetest-Tests
# ──────────────────────────────────────────────

class TestScoreReadingTest:

    def test_score_is_within_range(self):
        result = score_reading_test(anstrengung=3, unschaerfe=2, komfort=8)
        assert 0 <= result["score"] <= 100

    def test_low_effort_high_comfort_yields_good_score(self):
        result = score_reading_test(anstrengung=1, unschaerfe=1, komfort=9)
        assert result["score"] >= 50

    def test_high_effort_low_comfort_yields_lower_score(self):
        result_bad = score_reading_test(anstrengung=10, unschaerfe=10, komfort=0)
        result_good = score_reading_test(anstrengung=1, unschaerfe=1, komfort=10)
        assert result_bad["score"] < result_good["score"]

    def test_result_has_required_keys(self):
        result = score_reading_test(anstrengung=5, unschaerfe=5, komfort=5)
        assert "score" in result
        assert "label" in result
        assert "details" in result

    def test_details_contain_inputs(self):
        result = score_reading_test(anstrengung=4, unschaerfe=3, komfort=7)
        assert result["details"]["anstrengung"] == 4
        assert result["details"]["unschaerfe"] == 3
        assert result["details"]["komfort"] == 7

    @pytest.mark.parametrize("anstrengung,unschaerfe,komfort", [
        (0, 0, 0),
        (10, 10, 10),
        (0, 0, 10),
        (10, 10, 0),
    ])
    def test_boundary_values_stay_in_range(self, anstrengung, unschaerfe, komfort):
        result = score_reading_test(
            anstrengung=anstrengung,
            unschaerfe=unschaerfe,
            komfort=komfort,
        )
        assert 0 <= result["score"] <= 100


# ──────────────────────────────────────────────
# Label-Tests
# ──────────────────────────────────────────────

class TestLabels:

    def test_high_score_label(self):
        result = score_fixation_test(brightness=130.0, contrast=60.0)
        if result["score"] >= 80:
            assert result["label"] == "Sehr gut"

    def test_all_labels_are_german(self):
        """Stellt sicher, dass deutsche Labels verwendet werden."""
        german_labels = {"Sehr gut", "Gut", "Mittel", "Schwach", "Ungenügend"}

        for brightness in [20, 80, 130, 180, 230]:
            for contrast in [5, 20, 40, 60]:
                result = score_fixation_test(
                    brightness=float(brightness),
                    contrast=float(contrast),
                )
                assert result["label"] in german_labels, (
                    f"Unbekanntes Label: {result['label']}"
                )


# ──────────────────────────────────────────────
# v0.2: Kein-Gesicht-Tests
# ──────────────────────────────────────────────

class TestScoreFixationNoFace:

    def test_score_is_zero(self):
        result = score_fixation_no_face()
        assert result["score"] == 0.0

    def test_label_is_invalid(self):
        result = score_fixation_no_face()
        assert result["label"] == LABEL_INVALID

    def test_details_contain_grund(self):
        result = score_fixation_no_face()
        assert "grund" in result["details"]

    def test_result_has_required_keys(self):
        result = score_fixation_no_face()
        assert "score" in result
        assert "label" in result
        assert "details" in result

    def test_score_is_within_range(self):
        result = score_fixation_no_face()
        assert 0 <= result["score"] <= 100
