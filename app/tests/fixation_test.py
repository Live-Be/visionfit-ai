"""Fixationsstabilitäts-Test – kamerabasiert."""

import streamlit as st

from app.cv.image_utils import uploaded_file_to_bgr, bgr_to_gray
from app.cv.metrics import calculate_brightness_score, calculate_contrast_score
from app.scoring.rules import score_fixation_test, ScoreResult
from app.ui.components import show_score_card, show_section_header


def run_fixation_test() -> ScoreResult | None:
    """Führt den Fixationstest durch und gibt das Ergebnis zurück.

    Returns:
        ScoreResult wenn ein Bild aufgenommen und ausgewertet wurde, sonst None.
    """
    show_section_header("Fixationsstabilitäts-Test", "")

    st.write(
        "Schauen Sie direkt in die Kamera und halten Sie Ihren Blick auf den Mittelpunkt "
        "des Bildes fixiert. Nehmen Sie dann ein Foto auf."
    )

    st.markdown(
        """
        **Anleitung:**
        1. Positionieren Sie Ihr Gesicht mittig vor der Kamera
        2. Fokussieren Sie Ihren Blick auf einen festen Punkt
        3. Klicken Sie auf **Foto aufnehmen**
        """
    )

    camera_image = st.camera_input("Kamera-Aufnahme für Fixationstest")

    if camera_image is not None:
        with st.spinner("Bild wird analysiert…"):
            img_bgr = uploaded_file_to_bgr(camera_image)
            gray = bgr_to_gray(img_bgr)

            brightness = calculate_brightness_score(gray)
            contrast = calculate_contrast_score(gray)

        result = score_fixation_test(brightness=brightness, contrast=contrast)

        st.success("Analyse abgeschlossen!")
        show_score_card(
            label=result["label"],
            score=result["score"],
            details=result["details"],
        )

        return result

    return None
