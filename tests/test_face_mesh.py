"""pytest-Tests für das face_mesh Modul (v0.2)."""

import numpy as np
import pytest

from app.cv.face_mesh import (
    init_face_mesh,
    detect_face_landmarks,
    draw_face_landmarks,
    get_landmark_xy,
)


def _black_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Hilfsfunktion: Erzeugt ein schwarzes BGR-Testbild."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestInitFaceMesh:

    def test_returns_face_mesh_object(self):
        fm = init_face_mesh()
        assert fm is not None
        fm.close()

    def test_custom_params(self):
        fm = init_face_mesh(max_num_faces=2, min_detection_confidence=0.7)
        assert fm is not None
        fm.close()


class TestDetectFaceLandmarks:

    def test_returns_dict_with_required_keys(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        assert "face_detected" in result
        assert "landmark_count" in result
        assert "landmarks" in result
        assert "raw_result" in result

    def test_black_image_no_face_detected(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        assert result["face_detected"] is False
        assert result["landmark_count"] == 0
        assert result["landmarks"] == []

    def test_no_face_landmark_count_is_zero(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        assert result["landmark_count"] == 0

    def test_landmarks_list_empty_when_no_face(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        assert isinstance(result["landmarks"], list)
        assert len(result["landmarks"]) == 0

    def test_accepts_custom_face_mesh(self):
        fm = init_face_mesh()
        img = _black_image()
        result = detect_face_landmarks(img, face_mesh=fm)
        assert "face_detected" in result
        fm.close()

    def test_result_face_detected_is_bool(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        assert isinstance(result["face_detected"], bool)

    def test_result_landmark_count_is_int(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        assert isinstance(result["landmark_count"], int)


class TestDrawFaceLandmarks:

    def test_returns_ndarray(self):
        img = _black_image()
        result = detect_face_landmarks(img)
        annotated = draw_face_landmarks(img, result)
        assert isinstance(annotated, np.ndarray)

    def test_output_same_shape_as_input(self):
        img = _black_image(200, 300)
        result = detect_face_landmarks(img)
        annotated = draw_face_landmarks(img, result)
        assert annotated.shape == img.shape

    def test_does_not_modify_original(self):
        img = _black_image()
        original_copy = img.copy()
        result = detect_face_landmarks(img)
        draw_face_landmarks(img, result)
        np.testing.assert_array_equal(img, original_copy)

    def test_no_face_draws_red_border(self):
        img = _black_image(200, 200)
        result = {"face_detected": False, "landmark_count": 0, "landmarks": [], "raw_result": None}
        annotated = draw_face_landmarks(img, result)
        # Roter Rahmen: irgendwo sollte ein roter Pixel sein (B=0, G=0, R>0)
        red_pixels = (annotated[:, :, 2] > 150) & (annotated[:, :, 0] < 50)
        assert red_pixels.any(), "Kein roter Rahmen bei nicht erkanntem Gesicht"


class TestGetLandmarkXY:

    def test_returns_tuple(self):
        landmarks = [(0.5, 0.5, 0.0), (0.1, 0.2, 0.0)]
        xy = get_landmark_xy(landmarks, 0, (100, 100, 3))
        assert isinstance(xy, tuple)
        assert len(xy) == 2

    def test_normalized_center_maps_to_pixel_center(self):
        landmarks = [(0.5, 0.5, 0.0)]
        x, y = get_landmark_xy(landmarks, 0, (100, 200, 3))
        assert x == 100  # 0.5 * 200
        assert y == 50   # 0.5 * 100

    def test_zero_coords(self):
        landmarks = [(0.0, 0.0, 0.0)]
        x, y = get_landmark_xy(landmarks, 0, (480, 640, 3))
        assert x == 0
        assert y == 0
