"""Wiederverwendbare UI-Komponenten für die VisionFit AI Streamlit-App."""

import streamlit as st


def show_disclaimer() -> None:
    """Zeigt den gesetzlichen Disclaimer als Info-Box an."""
    st.info(
        "**Hinweis:** VisionFit AI ist kein Medizinprodukt und kein Diagnosetool. "
        "Dieser Prototyp dient ausschließlich Forschungs- und UX-Zwecken. "
        "Bitte konsultieren Sie bei gesundheitlichen Fragen einen Augenarzt oder Optiker."
    )


def show_score_card(label: str, score: float, details: dict) -> None:
    """Zeigt eine Score-Ergebniskarte an.

    Args:
        label:   Bewertungslabel (z.B. 'Gut', 'Sehr gut').
        score:   Numerischer Score (0–100).
        details: Detailwerte als Dictionary.
    """
    color = _score_color(score)
    st.markdown(
        f"""
        <div style="
            background: {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
            margin: 10px 0;
        ">
            <h2 style="margin:0; font-size: 3em;">{score:.0f}</h2>
            <p style="margin:4px 0; font-size: 1.3em; font-weight: bold;">{label}</p>
            <p style="margin:0; font-size: 0.9em; opacity: 0.85;">Score (0–100)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Details anzeigen"):
        for key, value in details.items():
            st.write(f"**{key}:** {value}")


def show_section_header(title: str, icon: str = "") -> None:
    """Zeigt einen Abschnitts-Header an.

    Args:
        title: Titel des Abschnitts.
        icon:  Optionales Emoji-Icon.
    """
    st.markdown(f"### {icon} {title}" if icon else f"### {title}")
    st.markdown("---")


def _score_color(score: float) -> str:
    """Gibt eine Hintergrundfarbe basierend auf dem Score zurück."""
    if score >= 80:
        return "#2ecc71"   # Grün
    if score >= 60:
        return "#3498db"   # Blau
    if score >= 40:
        return "#f39c12"   # Orange
    return "#e74c3c"       # Rot
