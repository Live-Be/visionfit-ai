"""pytest-Tests für das eye_metrics Modul (v0.2 Phase 3)."""

import math
import pytest

from app.cv.eye_metrics import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    DEFAULT_BLINK_THRESHOLD,
    DEFAULT_MIN_BLINK_FRAMES,
    MIN_RELIABLE_FRAMES,
    EAR_NORMAL_LOW,
    EAR_NORMAL_HIGH,
    eye_aspect_ratio,
    detect_blink,
    blink_rate,
    label_blink_rate,
    summarize_eye_metrics,
    _count_blink_events,
    _euclidean,
)


# ──────────────────────────────────────────────────────────────
# Hilfsfunktionen für Testdaten
# ──────────────────────────────────────────────────────────────

# Bildgröße für alle Tests
_SHAPE = (480, 640, 3)
_H, _W = 480, 640


def _norm(x_px: float, y_px: float) -> tuple[float, float, float]:
    """Konvertiert Pixel-Koordinaten in normierte (x, y, z=0)."""
    return (x_px / _W, y_px / _H, 0.0)


def _make_open_eye_landmarks(max_idx: int = 478) -> list:
    """Erstellt eine synthetische Landmark-Liste mit einem klar geöffneten Auge.

    Geometrie des rechten Auges (Indizes 33, 160, 158, 133, 153, 144):
    - p1 (33)  = linker Außenwinkel:     (100, 200) → EAR-Breite
    - p2 (160) = obere Außenseite:       (120, 185) → Oberlid außen
    - p3 (158) = obere Innenseite:       (160, 185) → Oberlid innen
    - p4 (133) = rechter Innenwinkel:    (180, 200) → EAR-Breite
    - p5 (153) = untere Innenseite:      (160, 215) → Unterlid innen
    - p6 (144) = untere Außenseite:      (120, 215) → Unterlid außen

    EAR = (d(p2,p6) + d(p3,p5)) / (2 * d(p1,p4))
        = (d((120,185),(120,215)) + d((160,185),(160,215))) / (2 * d((100,200),(180,200)))
        = (30 + 30) / (2 * 80) = 60/160 = 0.375  → offenes Auge
    """
    lm = [_norm(320.0, 240.0)] * max_idx  # neutrale Füllwerte

    # Rechtes Auge – geöffnet (EAR ≈ 0.375)
    lm[33]  = _norm(100.0, 200.0)  # p1 außen
    lm[160] = _norm(120.0, 185.0)  # p2 oben außen
    lm[158] = _norm(160.0, 185.0)  # p3 oben innen
    lm[133] = _norm(180.0, 200.0)  # p4 innen
    lm[153] = _norm(160.0, 215.0)  # p5 unten innen
    lm[144] = _norm(120.0, 215.0)  # p6 unten außen

    # Linkes Auge – geöffnet (gleiche Geometrie, gespiegelt, EAR ≈ 0.375)
    lm[362] = _norm(460.0, 200.0)  # p1 außen
    lm[385] = _norm(480.0, 185.0)  # p2 oben außen
    lm[387] = _norm(520.0, 185.0)  # p3 oben innen
    lm[263] = _norm(540.0, 200.0)  # p4 innen
    lm[373] = _norm(520.0, 215.0)  # p5 unten innen
    lm[380] = _norm(480.0, 215.0)  # p6 unten außen

    return lm


def _make_closed_eye_landmarks(max_idx: int = 478) -> list:
    """Erstellt eine synthetische Landmark-Liste mit einem geschlossenen Auge (Blink).

    EAR ≈ 0.025 (vertikaler Spalt nur 2px bei 80px Breite).
    """
    lm = _make_open_eye_landmarks(max_idx)

    # Rechtes Auge – fast geschlossen (EAR ≈ 0.0125)
    lm[33]  = _norm(100.0, 200.0)  # p1 außen
    lm[160] = _norm(120.0, 199.0)  # p2 oben außen  (nur 1px über Mitte)
    lm[158] = _norm(160.0, 199.0)  # p3 oben innen
    lm[133] = _norm(180.0, 200.0)  # p4 innen
    lm[153] = _norm(160.0, 201.0)  # p5 unten innen (nur 1px unter Mitte)
    lm[144] = _norm(120.0, 201.0)  # p6 unten außen

    # Linkes Auge – fast geschlossen (gleiche Geometrie)
    lm[362] = _norm(460.0, 200.0)
    lm[385] = _norm(480.0, 199.0)
    lm[387] = _norm(520.0, 199.0)
    lm[263] = _norm(540.0, 200.0)
    lm[373] = _norm(520.0, 201.0)
    lm[380] = _norm(480.0, 201.0)

    return lm


def _open_sequence(n: int = 50) -> list[list]:
    """Landmark-Sequenz mit immer offenen Augen."""
    return [_make_open_eye_landmarks() for _ in range(n)]


def _blink_sequence(n: int = 50, blink_at: int = 10, blink_len: int = 3) -> list[list]:
    """Landmark-Sequenz mit einem eingebetteten Blink."""
    frames = _open_sequence(n)
    for i in range(blink_at, min(blink_at + blink_len, n)):
        frames[i] = _make_closed_eye_landmarks()
    return frames


def _multi_blink_sequence(n: int = 90, blinks: int = 3) -> list[list]:
    """Sequenz mit mehreren Blinks gleichmäßig verteilt."""
    frames = _open_sequence(n)
    spacing = n // (blinks + 1)
    for b in range(blinks):
        start = spacing * (b + 1)
        for i in range(start, min(start + 3, n)):
            frames[i] = _make_closed_eye_landmarks()
    return frames


# ──────────────────────────────────────────────────────────────
# Interne Hilfsfunktionen
# ──────────────────────────────────────────────────────────────

class TestEuclidean:

    def test_same_point_is_zero(self):
        assert _euclidean((5, 5), (5, 5)) == 0.0

    def test_horizontal_distance(self):
        assert _euclidean((0, 0), (3, 0)) == pytest.approx(3.0)

    def test_vertical_distance(self):
        assert _euclidean((0, 0), (0, 4)) == pytest.approx(4.0)

    def test_diagonal_345(self):
        assert _euclidean((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_negative_coords(self):
        assert _euclidean((-1, -1), (2, 3)) == pytest.approx(5.0)


class TestCountBlinkEvents:

    def test_empty_returns_zero(self):
        assert _count_blink_events([]) == 0

    def test_no_blink_above_threshold(self):
        assert _count_blink_events([0.30, 0.32, 0.28]) == 0

    def test_single_long_blink(self):
        ears = [0.30] * 5 + [0.10] * 3 + [0.30] * 5
        assert _count_blink_events(ears) == 1

    def test_two_distinct_blinks(self):
        ears = [0.30] * 5 + [0.10] * 3 + [0.30] * 5 + [0.10] * 3 + [0.30] * 5
        assert _count_blink_events(ears) == 2

    def test_run_too_short_not_counted(self):
        ears = [0.30] * 5 + [0.10] * 1 + [0.30] * 5  # nur 1 Frame < threshold
        assert _count_blink_events(ears, min_frames=2) == 0

    def test_exactly_min_frames_counted(self):
        ears = [0.30] * 3 + [0.10] * 2 + [0.30] * 3
        assert _count_blink_events(ears, min_frames=2) == 1


# ──────────────────────────────────────────────────────────────
# eye_aspect_ratio
# ──────────────────────────────────────────────────────────────

class TestEyeAspectRatio:

    def test_returns_float(self):
        lm = _make_open_eye_landmarks()
        result = eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE)
        assert isinstance(result, float)

    def test_open_eye_high_ear(self):
        lm = _make_open_eye_landmarks()
        ear = eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE)
        assert ear > DEFAULT_BLINK_THRESHOLD, f"EAR {ear} sollte > {DEFAULT_BLINK_THRESHOLD}"

    def test_closed_eye_low_ear(self):
        lm = _make_closed_eye_landmarks()
        ear = eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE)
        assert ear < DEFAULT_BLINK_THRESHOLD, f"EAR {ear} sollte < {DEFAULT_BLINK_THRESHOLD}"

    def test_open_eye_known_value(self):
        # Aus _make_open_eye_landmarks: erwartet EAR ≈ 0.375
        lm = _make_open_eye_landmarks()
        ear = eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE)
        assert ear == pytest.approx(0.375, abs=0.01)

    def test_empty_landmarks_returns_zero(self):
        assert eye_aspect_ratio([], RIGHT_EYE_INDICES, _SHAPE) == 0.0

    def test_wrong_indices_count_returns_zero(self):
        lm = _make_open_eye_landmarks()
        assert eye_aspect_ratio(lm, [33, 160], _SHAPE) == 0.0  # nur 2 statt 6

    def test_out_of_range_index_returns_zero(self):
        # Sehr kurze Liste – alle HIGH_EYE_INDICES (33, 133, 144, ...) sind out of range
        lm = [_norm(320.0, 240.0)] * 10
        assert eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE) == 0.0

    def test_left_eye_same_geometry_as_right(self):
        lm = _make_open_eye_landmarks()
        ear_left = eye_aspect_ratio(lm, LEFT_EYE_INDICES, _SHAPE)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE)
        assert ear_left == pytest.approx(ear_right, abs=0.01)

    def test_ear_non_negative(self):
        for lm in [_make_open_eye_landmarks(), _make_closed_eye_landmarks()]:
            assert eye_aspect_ratio(lm, RIGHT_EYE_INDICES, _SHAPE) >= 0.0


# ──────────────────────────────────────────────────────────────
# detect_blink
# ──────────────────────────────────────────────────────────────

class TestDetectBlink:

    def test_empty_history_no_blink(self):
        assert detect_blink([]) is False

    def test_open_eyes_no_blink(self):
        ears = [0.30, 0.32, 0.28, 0.31, 0.29]
        assert detect_blink(ears) is False

    def test_single_frame_below_threshold_min2_no_blink(self):
        ears = [0.30, 0.10, 0.30]  # Nur 1 Frame unter threshold, min=2
        assert detect_blink(ears, min_frames=2) is False

    def test_two_frames_below_threshold_is_blink(self):
        ears = [0.30, 0.10, 0.10, 0.30]  # 2 Frames unter threshold
        assert detect_blink(ears, min_frames=2) is True

    def test_clear_blink_detected(self):
        ears = [0.30] * 5 + [0.12] * 3 + [0.30] * 5
        assert detect_blink(ears) is True

    def test_returns_bool(self):
        result = detect_blink([0.25, 0.30])
        assert isinstance(result, bool)

    def test_custom_threshold(self):
        # Niedrigerer Threshold → kein Blink erkannt
        ears = [0.15, 0.14, 0.15]  # unter 0.20 aber nicht unter 0.10
        assert detect_blink(ears, threshold=0.10) is False
        assert detect_blink(ears, threshold=0.20) is True

    @pytest.mark.parametrize("ears", [
        [0.30] * 10,                     # normal offen → kein Blink
        [0.25] * 5 + [0.26] * 5,         # über threshold → kein Blink
    ])
    def test_no_blink_cases(self, ears):
        assert detect_blink(ears) is False


# ──────────────────────────────────────────────────────────────
# blink_rate
# ──────────────────────────────────────────────────────────────

class TestBlinkRate:

    def test_empty_returns_zero(self):
        assert blink_rate([], fps=30.0) == 0.0

    def test_zero_fps_returns_zero(self):
        assert blink_rate([True, False, True], fps=0.0) == 0.0

    def test_negative_fps_returns_zero(self):
        assert blink_rate([True, False], fps=-1.0) == 0.0

    def test_no_blinks_returns_zero(self):
        events = [False] * 30
        assert blink_rate(events, fps=30.0) == 0.0

    def test_one_blink_in_one_second(self):
        # 30 Frames, 1 Blink-Ereignis → 60 Blinks/min
        events = [False] * 14 + [True, True] + [False] * 14
        rate = blink_rate(events, fps=30.0)
        assert rate == pytest.approx(60.0, abs=5.0)

    def test_returns_float(self):
        assert isinstance(blink_rate([False, True, False], fps=30.0), float)

    def test_rate_non_negative(self):
        for events in [[False] * 10, [True] * 5, [True, False] * 5]:
            assert blink_rate(events, fps=30.0) >= 0.0

    def test_multiple_blinks_proportional(self):
        # 3 Blink-Ereignisse in 90 Frames bei 30 fps = 3 Sekunden → 60 Blinks/min
        events = []
        for _ in range(3):
            events += [False] * 12 + [True, True] + [False] * 14
        rate = blink_rate(events, fps=30.0)
        assert rate == pytest.approx(60.0, abs=10.0)


# ──────────────────────────────────────────────────────────────
# label_blink_rate
# ──────────────────────────────────────────────────────────────

class TestLabelBlinkRate:

    _VALID_LABELS = {
        "Kein Blinken erkannt",
        "Sehr selten",
        "Unterdurchschnittlich",
        "Normal",
        "Erhöht",
        "Sehr häufig",
    }

    def test_zero_rate(self):
        assert label_blink_rate(0.0) == "Kein Blinken erkannt"

    def test_normal_range(self):
        for rate in [10.0, 15.0, 20.0, 24.9]:
            assert label_blink_rate(rate) == "Normal"

    def test_very_rare(self):
        assert label_blink_rate(2.0) == "Sehr selten"

    def test_below_average(self):
        assert label_blink_rate(7.0) == "Unterdurchschnittlich"

    def test_elevated(self):
        assert label_blink_rate(30.0) == "Erhöht"

    def test_very_frequent(self):
        assert label_blink_rate(50.0) == "Sehr häufig"

    @pytest.mark.parametrize("rate", [0, 3, 8, 15, 30, 50])
    def test_all_rates_return_valid_label(self, rate):
        assert label_blink_rate(float(rate)) in self._VALID_LABELS

    def test_labels_are_german(self):
        for rate in range(0, 60, 5):
            label = label_blink_rate(float(rate))
            assert isinstance(label, str) and len(label) > 0


# ──────────────────────────────────────────────────────────────
# summarize_eye_metrics (Integrations-Tests)
# ──────────────────────────────────────────────────────────────

class TestSummarizeEyeMetrics:

    def test_returns_dict(self):
        result = summarize_eye_metrics(_open_sequence(50), _SHAPE)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = summarize_eye_metrics(_open_sequence(5), _SHAPE)
        required = {
            "ear_mean", "ear_std", "blink_count", "blink_rate",
            "blink_events", "ear_history", "frame_count",
            "is_reliable", "label", "warnung",
        }
        assert required.issubset(result.keys())

    def test_empty_input_defensive(self):
        result = summarize_eye_metrics([], _SHAPE)
        assert result["frame_count"] == 0
        assert result["is_reliable"] is False
        assert result["ear_mean"] == 0.0
        assert result["blink_count"] == 0
        assert result["warnung"] is not None

    def test_single_frame_not_reliable(self):
        result = summarize_eye_metrics(_open_sequence(1), _SHAPE)
        assert result["is_reliable"] is False
        assert result["warnung"] is not None

    def test_reliable_with_enough_frames(self):
        result = summarize_eye_metrics(_open_sequence(MIN_RELIABLE_FRAMES), _SHAPE)
        assert result["is_reliable"] is True
        assert result["warnung"] is None

    def test_open_eyes_no_blink(self):
        result = summarize_eye_metrics(_open_sequence(50), _SHAPE)
        assert result["blink_count"] == 0

    def test_blink_sequence_detects_blink(self):
        result = summarize_eye_metrics(
            _blink_sequence(n=50, blink_at=10, blink_len=3), _SHAPE
        )
        assert result["blink_count"] >= 1

    def test_open_ear_mean_above_threshold(self):
        result = summarize_eye_metrics(_open_sequence(50), _SHAPE)
        assert result["ear_mean"] > DEFAULT_BLINK_THRESHOLD

    def test_blink_sequence_ear_mean_lower_than_open(self):
        r_open = summarize_eye_metrics(_open_sequence(50), _SHAPE)
        # Mehr Blink-Frames → niedrigerer Durchschnitts-EAR
        r_blink = summarize_eye_metrics(
            _blink_sequence(n=50, blink_at=5, blink_len=20), _SHAPE
        )
        assert r_blink["ear_mean"] < r_open["ear_mean"]

    def test_ear_history_length_matches_frame_count(self):
        result = summarize_eye_metrics(_open_sequence(20), _SHAPE)
        assert len(result["ear_history"]) == result["frame_count"]

    def test_blink_events_length_matches_frame_count(self):
        result = summarize_eye_metrics(_open_sequence(20), _SHAPE)
        assert len(result["blink_events"]) == result["frame_count"]

    def test_blink_events_are_bools(self):
        result = summarize_eye_metrics(_open_sequence(10), _SHAPE)
        for ev in result["blink_events"]:
            assert isinstance(ev, bool)

    def test_blink_rate_non_negative(self):
        for seq in [_open_sequence(50), _blink_sequence(50)]:
            result = summarize_eye_metrics(seq, _SHAPE)
            assert result["blink_rate"] >= 0.0

    def test_multiple_blinks_counted(self):
        result = summarize_eye_metrics(_multi_blink_sequence(n=90, blinks=3), _SHAPE)
        assert result["blink_count"] == 3

    def test_warnung_none_when_reliable(self):
        result = summarize_eye_metrics(_open_sequence(MIN_RELIABLE_FRAMES), _SHAPE)
        assert result["warnung"] is None

    def test_warnung_set_when_few_frames(self):
        result = summarize_eye_metrics(_open_sequence(5), _SHAPE)
        assert isinstance(result["warnung"], str) and len(result["warnung"]) > 0

    def test_label_is_string(self):
        result = summarize_eye_metrics(_open_sequence(5), _SHAPE)
        assert isinstance(result["label"], str)

    def test_skips_empty_frames(self):
        seq = _open_sequence(10)
        seq_with_empty = seq[:5] + [[]] + seq[5:]
        result = summarize_eye_metrics(seq_with_empty, _SHAPE)
        # Leerer Frame wird übersprungen → frame_count = 10
        assert result["frame_count"] == 10

    def test_custom_fps_affects_blink_rate(self):
        seq = _blink_sequence(n=30, blink_at=5, blink_len=3)
        r_30 = summarize_eye_metrics(seq, _SHAPE, fps=30.0)
        r_10 = summarize_eye_metrics(seq, _SHAPE, fps=10.0)
        # Gleiche Blink-Anzahl, aber unterschiedliche Rate (verschiedene Dauer)
        assert r_10["blink_rate"] != r_30["blink_rate"]

    def test_ear_values_in_ear_history_non_negative(self):
        result = summarize_eye_metrics(_open_sequence(20), _SHAPE)
        for ear in result["ear_history"]:
            assert ear >= 0.0


# ──────────────────────────────────────────────────────────────
# Konstanten-Tests
# ──────────────────────────────────────────────────────────────

class TestConstants:

    def test_left_eye_indices_length(self):
        assert len(LEFT_EYE_INDICES) == 6

    def test_right_eye_indices_length(self):
        assert len(RIGHT_EYE_INDICES) == 6

    def test_blink_threshold_reasonable(self):
        assert 0.10 < DEFAULT_BLINK_THRESHOLD < 0.30

    def test_min_blink_frames_positive(self):
        assert DEFAULT_MIN_BLINK_FRAMES >= 1

    def test_min_reliable_frames_meaningful(self):
        assert MIN_RELIABLE_FRAMES >= 10

    def test_ear_normal_range_sensible(self):
        assert EAR_NORMAL_LOW < EAR_NORMAL_HIGH
        assert 0.0 < EAR_NORMAL_LOW
        assert EAR_NORMAL_HIGH < 1.0
