"""Fixationsstabilitäts-Test – kamerabasiert mit MediaPipe Face Landmarks (v0.2).

v0.2 Phase 2 Vorbereitung:
    head_stability.summarize_head_stability() ist für die Multi-Frame-Videoanalyse
    (v0.3) vorbereitet. Im Einzelbild-Modus wird der Stabilitätsscore nicht berechnet.
    Der Scoring-Aufruf kann bei Bedarf auf score_fixation_with_stability() umgestellt
    werden sobald Video-Input verfügbar ist.
"""

import cv2
import streamlit as st

from app.cv.image_utils import uploaded_file_to_bgr, bgr_to_gray
from app.cv.metrics import calculate_brightness_score, calculate_contrast_score
from app.cv.face_mesh import detect_face_landmarks, draw_face_landmarks
from app.scoring.rules import score_fixation_test, score_fixation_no_face, ScoreResult
from app.ui.components import show_score_card, show_section_header


def run_fixation_test() -> ScoreResult | None:
    """Führt den Fixationstest durch und gibt das Ergebnis zurück.

    v0.2: Erweitert um MediaPipe Face Landmark Erkennung.
    Ohne erkanntes Gesicht wird kein Score berechnet.

    Returns:
        ScoreResult wenn ein Bild aufgenommen und ausgewertet wurde, sonst None.
    """
    show_section_header("Fixationsstabilitäts-Test", "")

    st.info(
        " **Bitte schauen Sie direkt in die Kamera.**  \n"
        "Halten Sie Ihren Blick auf den Mittelpunkt des Bildes fixiert "
        "und sorgen Sie für gute Beleuchtung."
    )

    st.markdown(
        """
        **Anleitung:**
        1. Positionieren Sie Ihr Gesicht mittig vor der Kamera
        2. Sorgen Sie für ausreichende Beleuchtung (kein Gegenlicht)
        3. Schauen Sie direkt in die Kamera
        4. Klicken Sie auf **Foto aufnehmen**
        """
    )

    camera_image = st.camera_input("Kamera-Aufnahme für Fixationstest")

    if camera_image is None:
        return None

    with st.spinner("Bild wird analysiert…"):
        img_bgr = uploaded_file_to_bgr(camera_image)
        gray = bgr_to_gray(img_bgr)

        brightness = calculate_brightness_score(gray)
        contrast = calculate_contrast_score(gray)

        landmarks_result = detect_face_landmarks(img_bgr)

    # ── Gesichtsstatus anzeigen ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Analyseergebnis")

    col_status, col_count = st.columns(2)

    with col_status:
        if landmarks_result["face_detected"]:
            st.success(" Gesicht erkannt")
        else:
            st.error(" Gesicht konnte nicht erkannt werden")

    with col_count:
        st.metric(
            label="Erkannte Landmarks",
            value=landmarks_result["landmark_count"],
        )

    # ── Annotiertes Bild anzeigen ────────────────────────────────────
    annotated_bgr = draw_face_landmarks(img_bgr, landmarks_result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    st.image(
        annotated_rgb,
        caption="Erkannte Gesichts-Landmarks" if landmarks_result["face_detected"]
                else "Kein Gesicht erkannt",
        use_container_width=True,
    )

    # ── Kein Gesicht → ungültiger Test ──────────────────────────────
    if not landmarks_result["face_detected"]:
        st.warning(
            " **Hinweis:** Das Gesicht konnte im Bild nicht erkannt werden.  \n"
            "Bitte stellen Sie sicher, dass:  \n"
            "- Ihr Gesicht vollständig sichtbar ist  \n"
            "- Die Beleuchtung ausreichend ist  \n"
            "- Kein starkes Gegenlicht vorhanden ist  \n\n"
            "Nehmen Sie ein neues Foto auf, um den Test fortzusetzen."
        )
        result = score_fixation_no_face()
        show_score_card(
            label=result["label"],
            score=result["score"],
            details=result["details"],
        )
        return result

    # ── Scoring ──────────────────────────────────────────────────────
    st.success("Analyse abgeschlossen – Score wird berechnet…")
    result = score_fixation_test(brightness=brightness, contrast=contrast)

    # Landmark-Info zu Details hinzufügen
    result["details"]["erkannte_landmarks"] = landmarks_result["landmark_count"]

    show_score_card(
        label=result["label"],
        score=result["score"],
        details=result["details"],
    )

    return result
