"""MediaPipe Face Mesh Integration für VisionFit AI v0.2.

Stellt Funktionen zur Gesichtserkennung und Landmark-Extraktion bereit.
Vorbereitet für Erweiterungen: Head-Stability (Phase 2), Blink-Detection (Phase 3).
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Defensive MediaPipe-Initialisierung – mp.solutions ist auf manchen Plattformen
# (z. B. Streamlit Cloud / Linux ARM) nach dem Import noch nicht verfügbar.
try:
    import mediapipe as mp  # type: ignore

    logger.debug("mediapipe importiert: repr=%s, __file__=%s", repr(mp), getattr(mp, "__file__", "?"))

    _mp_face_mesh = mp.solutions.face_mesh
    _mp_drawing = mp.solutions.drawing_utils
    _mp_drawing_styles = mp.solutions.drawing_styles

    _MEDIAPIPE_AVAILABLE = True
    logger.debug("mp.solutions erfolgreich geladen (face_mesh, drawing_utils, drawing_styles)")

except Exception as _mp_init_error:  # noqa: BLE001
    _mp_face_mesh = None  # type: ignore[assignment]
    _mp_drawing = None  # type: ignore[assignment]
    _mp_drawing_styles = None  # type: ignore[assignment]
    _MEDIAPIPE_AVAILABLE = False
    logger.warning(
        "MediaPipe konnte nicht initialisiert werden – FaceMesh nicht verfügbar. "
        "Fehler: %s",
        _mp_init_error,
    )


def _require_mediapipe() -> None:
    """Wirft einen klaren RuntimeError wenn MediaPipe nicht verfügbar ist."""
    if not _MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            "MediaPipe ist auf dieser Plattform nicht verfügbar. "
            "mp.solutions konnte nicht geladen werden. "
            "Prüfe die Deployment-Logs auf den genauen Initialisierungsfehler."
        )


def init_face_mesh(
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
) -> object:
    """Initialisiert ein MediaPipe FaceMesh-Objekt.

    Args:
        static_image_mode:          True für Einzelbilder (kein Video-Tracking).
        max_num_faces:               Maximale Anzahl zu erkennender Gesichter.
        refine_landmarks:            Aktiviert Iris-Landmarks (478 statt 468).
        min_detection_confidence:    Mindestkonfidenz für Gesichtserkennung.

    Returns:
        Konfiguriertes FaceMesh-Objekt.
    """
    _require_mediapipe()
    return _mp_face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
    )


def detect_face_landmarks(
    image_bgr: np.ndarray,
    face_mesh: object | None = None,
) -> dict:
    """Erkennt Gesichts-Landmarks in einem BGR-Bild.

    Erstellt intern ein temporäres FaceMesh-Objekt wenn keines übergeben wird.
    Für wiederholte Aufrufe sollte ein wiederverwendetes FaceMesh übergeben werden.

    Args:
        image_bgr:  BGR-Bild als numpy-Array (H, W, 3).
        face_mesh:  Optional: bereits initialisiertes FaceMesh-Objekt.

    Returns:
        Dictionary mit:
            - face_detected (bool):    Ob ein Gesicht erkannt wurde.
            - landmark_count (int):    Anzahl erkannter Landmarks (0 wenn kein Gesicht).
            - landmarks (list):        Liste von (x, y, z)-Tupeln in normalisierten
                                       Koordinaten [0.0, 1.0]. Leer wenn kein Gesicht.
            - raw_result:              Rohe MediaPipe-Ergebnis-Struktur (für
                                       Weiterverarbeitung in Phase 2/3).
    """
    close_after = face_mesh is None
    if face_mesh is None:
        face_mesh = init_face_mesh()

    # MediaPipe erwartet RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    result = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True

    if close_after:
        face_mesh.close()

    if not result.multi_face_landmarks:
        return {
            "face_detected": False,
            "landmark_count": 0,
            "landmarks": [],
            "raw_result": result,
        }

    # Erstes Gesicht extrahieren
    face = result.multi_face_landmarks[0]
    landmarks = [
        (lm.x, lm.y, lm.z)
        for lm in face.landmark
    ]

    return {
        "face_detected": True,
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
        "raw_result": result,
    }


def draw_face_landmarks(
    image_bgr: np.ndarray,
    landmarks_result: dict,
    draw_tesselation: bool = True,
    draw_contours: bool = True,
    draw_irises: bool = True,
) -> np.ndarray:
    """Zeichnet Gesichts-Landmarks auf eine Bildkopie.

    Args:
        image_bgr:         Original BGR-Bild.
        landmarks_result:  Ergebnis-Dictionary von detect_face_landmarks().
        draw_tesselation:  Zeichnet die Gesichts-Tesselierung (Mesh).
        draw_contours:     Zeichnet Gesichtskonturen (Augen, Mund, etc.).
        draw_irises:       Zeichnet Iris-Landmarks (falls vorhanden).

    Returns:
        Annotiertes BGR-Bild als neue numpy-Array-Kopie.
    """
    _require_mediapipe()
    annotated = image_bgr.copy()

    if not landmarks_result["face_detected"]:
        # Roten Rahmen zeichnen wenn kein Gesicht erkannt
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (10, 10), (w - 10, h - 10), (0, 0, 200), 3)
        cv2.putText(
            annotated,
            "Kein Gesicht erkannt",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 200),
            2,
            cv2.LINE_AA,
        )
        return annotated

    raw_result = landmarks_result["raw_result"]

    for face_landmarks in raw_result.multi_face_landmarks:
        if draw_tesselation:
            _mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=_mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        if draw_contours:
            _mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=_mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

        if draw_irises:
            _mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=_mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

    return annotated


def get_landmark_xy(landmarks: list, index: int, image_shape: tuple) -> tuple[int, int]:
    """Gibt Pixel-Koordinaten eines einzelnen Landmarks zurück.

    Hilfsfunktion für Phase 2 (Head-Stability) und Phase 3 (Blink-Detection).

    Args:
        landmarks:    Liste von (x, y, z)-Tupeln (normalisiert 0–1).
        index:        Landmark-Index (0–477 bei refine_landmarks=True).
        image_shape:  Bildgröße als (height, width, ...).

    Returns:
        (x_pixel, y_pixel) als Integer-Tupel.
    """
    h, w = image_shape[:2]
    x, y, _ = landmarks[index]
    return int(x * w), int(y * h)
