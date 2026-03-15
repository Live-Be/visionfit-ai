"""Tests für v0.3 Video-Analyse-Pipeline.

Testet:
    - app.cv.video_capture:      validate_frames, build_frame_sequence,
                                 capture_frame_sequence
    - app.cv.landmark_pipeline:  extract_landmarks_from_frames
    - app.cv.video_analysis:     analyze_video_sequence
    - app.scoring.rules:         score_fixation_combined

Kein Streamlit-Import, keine reale Kamera erforderlich.
Synthetische Frames (schwarze / graue numpy Arrays) werden verwendet,
um die Logik ohne echte Kamera-Hardware zu testen.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.cv.video_capture import (
    MAX_FRAMES,
    validate_frames,
    build_frame_sequence,
    capture_frame_sequence,
)
from app.cv.landmark_pipeline import extract_landmarks_from_frames
from app.cv.video_analysis import (
    MIN_RELIABLE_DETECTION_RATE,
    analyze_video_sequence,
)
from app.scoring.rules import score_fixation_combined


# ──────────────────────────────────────────────────────────────────────────────
# Test-Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _black_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Erstellt einen schwarzen BGR-Frame (kein Gesicht erkennbar)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _gray_frame(val: int = 128, h: int = 480, w: int = 640) -> np.ndarray:
    """Erstellt einen grauen BGR-Frame mit definierter Helligkeit."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def _make_frames(n: int, val: int = 0) -> list[np.ndarray]:
    """Erstellt eine Liste von n grauen/schwarzen Frames."""
    return [_gray_frame(val) for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# 1. validate_frames
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateFrames:
    def test_empty_input_returns_empty(self):
        assert validate_frames([]) == []

    def test_none_filtered(self):
        assert validate_frames([None, None]) == []

    def test_valid_frame_kept(self):
        frames = [_black_frame()]
        assert len(validate_frames(frames)) == 1

    def test_multiple_valid_frames(self):
        frames = _make_frames(5)
        assert len(validate_frames(frames)) == 5

    def test_mixed_none_and_valid(self):
        frames = [None, _black_frame(), None, _black_frame()]
        assert len(validate_frames(frames)) == 2

    def test_non_ndarray_filtered(self):
        assert validate_frames(["not_an_array", 42, b"bytes"]) == []

    def test_wrong_shape_2d_filtered(self):
        bad = np.zeros((480, 640), dtype=np.uint8)  # 2D statt 3D
        assert validate_frames([bad]) == []

    def test_wrong_channels_filtered(self):
        bad = np.zeros((480, 640, 4), dtype=np.uint8)  # BGRA statt BGR
        assert validate_frames([bad]) == []

    def test_empty_array_filtered(self):
        bad = np.zeros((0, 0, 3), dtype=np.uint8)
        assert validate_frames([bad]) == []

    def test_returns_list(self):
        result = validate_frames([_black_frame()])
        assert isinstance(result, list)

    def test_valid_frames_are_ndarrays(self):
        result = validate_frames([_black_frame()])
        assert all(isinstance(f, np.ndarray) for f in result)


# ──────────────────────────────────────────────────────────────────────────────
# 2. build_frame_sequence
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildFrameSequence:
    def test_returns_dict(self):
        result = build_frame_sequence([])
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = build_frame_sequence([])
        assert {"frames", "fps", "frame_count"} <= result.keys()

    def test_empty_input_zero_count(self):
        result = build_frame_sequence([])
        assert result["frame_count"] == 0
        assert result["frames"] == []

    def test_valid_frame_counted(self):
        result = build_frame_sequence([_black_frame()])
        assert result["frame_count"] == 1

    def test_fps_preserved(self):
        result = build_frame_sequence([_black_frame()], fps=25.0)
        assert result["fps"] == 25.0

    def test_fps_below_one_clipped(self):
        result = build_frame_sequence([_black_frame()], fps=0.0)
        assert result["fps"] >= 1.0

    def test_fps_negative_clipped(self):
        result = build_frame_sequence([_black_frame()], fps=-5.0)
        assert result["fps"] >= 1.0

    def test_max_frames_truncated(self):
        frames = _make_frames(20)
        result = build_frame_sequence(frames, max_frames=10)
        assert result["frame_count"] == 10

    def test_truncation_keeps_newest_frames(self):
        # Letzter Frame ist weiß, Rest schwarz – bei Kürzung muss weißer Frame erhalten bleiben
        frames = _make_frames(10, val=0)
        frames.append(_gray_frame(255))
        result = build_frame_sequence(frames, max_frames=5)
        last_frame = result["frames"][-1]
        assert last_frame.mean() > 200  # weißer Frame

    def test_within_max_not_truncated(self):
        frames = _make_frames(50)
        result = build_frame_sequence(frames, max_frames=MAX_FRAMES)
        assert result["frame_count"] == 50

    def test_none_frames_filtered(self):
        frames = [None, _black_frame(), None]
        result = build_frame_sequence(frames)
        assert result["frame_count"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. capture_frame_sequence
# ──────────────────────────────────────────────────────────────────────────────

class TestCaptureFrameSequence:
    """Tests für capture_frame_sequence (kein Kamera-Zugriff erwartet in CI)."""

    def test_returns_dict(self):
        result = capture_frame_sequence(seconds=0.1)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = capture_frame_sequence(seconds=0.1)
        assert {"frames", "fps", "frame_count"} <= result.keys()

    def test_frames_is_list(self):
        result = capture_frame_sequence(seconds=0.1)
        assert isinstance(result["frames"], list)

    def test_frame_count_consistent(self):
        result = capture_frame_sequence(seconds=0.1)
        assert result["frame_count"] == len(result["frames"])

    def test_no_crash_on_invalid_camera(self):
        # Sehr hoher Index → keine Kamera vorhanden → sollte nicht crashen
        result = capture_frame_sequence(seconds=0.1, camera_index=99)
        assert result["frame_count"] == 0
        assert result["frames"] == []

    def test_fps_fallback_positive(self):
        result = capture_frame_sequence(seconds=0.1, fps=25.0, camera_index=99)
        assert result["fps"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 4. extract_landmarks_from_frames
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractLandmarksFromFrames:
    def test_empty_input_returns_empty(self):
        result = extract_landmarks_from_frames([])
        assert result["frames_landmarks"] == []
        assert result["face_detection_rate"] == 0.0

    def test_returns_dict_with_required_keys(self):
        result = extract_landmarks_from_frames([_black_frame()])
        assert {"frames_landmarks", "face_detection_rate"} <= result.keys()

    def test_black_frames_no_face(self):
        frames = _make_frames(5)
        result = extract_landmarks_from_frames(frames)
        assert result["face_detection_rate"] == 0.0

    def test_black_frames_landmarks_all_none(self):
        frames = _make_frames(3)
        result = extract_landmarks_from_frames(frames)
        assert all(lm is None for lm in result["frames_landmarks"])

    def test_length_matches_input(self):
        frames = _make_frames(7)
        result = extract_landmarks_from_frames(frames)
        assert len(result["frames_landmarks"]) == 7

    def test_detection_rate_zero_for_black_frames(self):
        frames = _make_frames(10)
        result = extract_landmarks_from_frames(frames)
        assert result["face_detection_rate"] == 0.0

    def test_detection_rate_is_float(self):
        result = extract_landmarks_from_frames([_black_frame()])
        assert isinstance(result["face_detection_rate"], float)

    def test_detection_rate_in_range(self):
        frames = _make_frames(5)
        result = extract_landmarks_from_frames(frames)
        assert 0.0 <= result["face_detection_rate"] <= 1.0

    def test_handles_none_frame_in_list(self):
        frames = [None, _black_frame(), None]
        result = extract_landmarks_from_frames(frames)
        assert result["frame_count"] if "frame_count" in result else True
        assert len(result["frames_landmarks"]) == 3

    def test_face_detection_rate_rounded(self):
        # Rate soll auf 3 Dezimalstellen gerundet sein
        frames = _make_frames(3)
        result = extract_landmarks_from_frames(frames)
        rate = result["face_detection_rate"]
        assert rate == round(rate, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 5. analyze_video_sequence
# ──────────────────────────────────────────────────────────────────────────────

class TestAnalyzeVideoSequence:
    def test_empty_frames_returns_dict(self):
        result = analyze_video_sequence([])
        assert isinstance(result, dict)

    def test_empty_frames_not_reliable(self):
        result = analyze_video_sequence([])
        assert result["is_reliable"] is False

    def test_empty_frames_zero_count(self):
        result = analyze_video_sequence([])
        assert result["frame_count"] == 0

    def test_has_all_required_keys(self):
        result = analyze_video_sequence([_black_frame()])
        expected = {
            "brightness", "contrast", "face_detection_rate",
            "head_stability_score", "head_stability_label", "head_stability_reliable",
            "blink_rate", "blink_count", "ear_mean", "ear_std",
            "blink_reliable", "frame_count", "is_reliable", "warnung",
        }
        assert expected <= result.keys()

    def test_black_frames_no_face_unreliable(self):
        frames = _make_frames(30)
        result = analyze_video_sequence(frames)
        assert result["face_detection_rate"] == 0.0
        assert result["is_reliable"] is False

    def test_black_frames_warnung_set(self):
        frames = _make_frames(10)
        result = analyze_video_sequence(frames)
        assert result["warnung"] is not None
        assert isinstance(result["warnung"], str)

    def test_brightness_in_range(self):
        frames = [_gray_frame(128)]
        result = analyze_video_sequence(frames)
        assert 0.0 <= result["brightness"] <= 255.0

    def test_contrast_non_negative(self):
        frames = [_black_frame()]
        result = analyze_video_sequence(frames)
        assert result["contrast"] >= 0.0

    def test_head_stability_score_in_range(self):
        frames = _make_frames(10)
        result = analyze_video_sequence(frames)
        assert 0.0 <= result["head_stability_score"] <= 100.0

    def test_blink_rate_non_negative(self):
        frames = _make_frames(30)
        result = analyze_video_sequence(frames)
        assert result["blink_rate"] >= 0.0

    def test_frame_count_matches_input(self):
        n = 15
        frames = _make_frames(n)
        result = analyze_video_sequence(frames)
        assert result["frame_count"] == n

    def test_bright_frame_higher_brightness(self):
        dark_frames = [_gray_frame(10)]
        bright_frames = [_gray_frame(200)]
        dark_result = analyze_video_sequence(dark_frames)
        bright_result = analyze_video_sequence(bright_frames)
        assert bright_result["brightness"] > dark_result["brightness"]

    def test_is_reliable_false_when_no_face(self):
        frames = _make_frames(90)  # Viele Frames, aber kein Gesicht
        result = analyze_video_sequence(frames)
        # Keine Gesichtserkennung → nicht zuverlässig
        assert result["is_reliable"] is False

    def test_warnung_german_text_when_unreliable(self):
        frames = _make_frames(5)
        result = analyze_video_sequence(frames)
        if result["warnung"] is not None:
            # Stichprobenartig prüfen dass Warnung Deutsch ist
            german_words = ["erkannt", "Frames", "Bitte", "mindestens", "sicher"]
            assert any(word in result["warnung"] for word in german_words)

    def test_fps_parameter_affects_blink_rate(self):
        """Unterschiedliche FPS → unterschiedliche Blinkraten-Berechnung."""
        frames = _make_frames(30)
        result_30 = analyze_video_sequence(frames, fps=30.0)
        result_10 = analyze_video_sequence(frames, fps=10.0)
        # Mit Schwarzframes: blink_rate ist in beiden Fällen 0 (kein EAR-Signal)
        # Aber die Funktion muss ohne Fehler ausführen
        assert isinstance(result_30["blink_rate"], float)
        assert isinstance(result_10["blink_rate"], float)


# ──────────────────────────────────────────────────────────────────────────────
# 6. score_fixation_combined
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreFixationCombined:
    def test_returns_score_result_keys(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=15.0,
        )
        assert {"score", "label", "details"} <= result.keys()

    def test_score_in_range(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0,
        )
        assert 0.0 <= result["score"] <= 100.0

    def test_score_is_float(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0,
        )
        assert isinstance(result["score"], float)

    def test_label_is_string(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0,
        )
        assert isinstance(result["label"], str)

    def test_none_blink_rate_no_penalty(self):
        """Keine Blinkrate → kein Abzug (blink_adj=0)."""
        r_none = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=None,
        )
        r_normal = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=15.0,
        )
        # Normale Blinkrate und keine Blinkrate sollten gleichen Score ergeben
        assert r_none["score"] == r_normal["score"]

    def test_extreme_blink_rate_reduces_score(self):
        """Sehr hohe Blinkrate soll Score leicht reduzieren."""
        r_normal = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=15.0,
        )
        r_extreme = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=60.0,
        )
        assert r_extreme["score"] < r_normal["score"]

    def test_blink_reduction_is_small(self):
        """Blink-Abzug soll maximal ~4 Punkte betragen."""
        r_best = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=15.0,
        )
        r_worst_blink = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=100.0,
        )
        reduction = r_best["score"] - r_worst_blink["score"]
        # Maximale Reduktion durch Blinken: 20% * (100-80) = 4 Punkte
        assert reduction <= 5.0  # leicht > 4 für Fließkomma-Toleranz

    def test_high_stability_increases_score(self):
        r_low = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=0.0,
        )
        r_high = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=100.0,
        )
        assert r_high["score"] > r_low["score"]

    def test_stability_weight_30_percent(self):
        """30 % Gewichtung der Stabilität prüfen."""
        # Gute Bildqualität (brightness=130, contrast=60 → image_score=100)
        r_0 = score_fixation_combined(brightness=130.0, contrast=60.0, head_stability_score=0.0)
        r_100 = score_fixation_combined(brightness=130.0, contrast=60.0, head_stability_score=100.0)
        # Unterschied sollte ~30 Punkte sein (30% von 100)
        diff = r_100["score"] - r_0["score"]
        assert 28.0 <= diff <= 32.0

    def test_details_contains_stability(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=75.0,
        )
        assert "head_stability_score" in result["details"]
        assert result["details"]["head_stability_score"] == 75.0

    def test_details_contains_blink_score(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=75.0, blink_rate=15.0,
        )
        assert "blink_score" in result["details"]

    def test_details_contains_gewichtung(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=75.0,
        )
        assert "gewichtung" in result["details"]
        assert "50%" in result["details"]["gewichtung"]
        assert "30%" in result["details"]["gewichtung"]
        assert "20%" in result["details"]["gewichtung"]

    def test_extreme_values_stay_in_range(self):
        """Extreme Eingaben dürfen Score nie aus [0, 100] treiben."""
        for brightness in [0.0, 255.0]:
            for contrast in [0.0, 128.0]:
                for stability in [0.0, 100.0]:
                    for blink_r in [None, 0.0, 100.0]:
                        r = score_fixation_combined(
                            brightness=brightness, contrast=contrast,
                            head_stability_score=stability, blink_rate=blink_r,
                        )
                        assert 0.0 <= r["score"] <= 100.0

    @pytest.mark.parametrize("blink_rate", [10.0, 15.0, 20.0, 25.0])
    def test_normal_blink_rate_no_penalty(self, blink_rate):
        """Normaler Blinkbereich (10–25/min) → kein Abzug."""
        r_no_blink = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=None,
        )
        r_with_blink = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=blink_rate,
        )
        assert r_with_blink["score"] == r_no_blink["score"]

    def test_blink_rate_none_stored_in_details(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=None,
        )
        assert result["details"]["blink_rate_pro_min"] is None

    def test_blink_rate_value_stored_in_details(self):
        result = score_fixation_combined(
            brightness=130.0, contrast=40.0,
            head_stability_score=80.0, blink_rate=18.5,
        )
        assert result["details"]["blink_rate_pro_min"] == 18.5
