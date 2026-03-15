"""Computer-Vision-Hilfsfunktionen für Bildumwandlungen."""

import numpy as np
import cv2


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Konvertiert ein Streamlit-UploadedFile-Objekt in ein BGR-NumPy-Array.

    Args:
        uploaded_file: Streamlit-UploadedFile (z.B. von st.camera_input).

    Returns:
        BGR-Array (H, W, 3) als np.ndarray.
    """
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr


def bgr_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Konvertiert ein BGR-Bild in Graustufen.

    Args:
        img_bgr: BGR-Array (H, W, 3).

    Returns:
        Graustufenbild (H, W) als np.ndarray.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
