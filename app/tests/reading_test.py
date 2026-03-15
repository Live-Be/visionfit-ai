"""Lese-Komfort-Test – Selbsteinschätzung via Slider."""

import streamlit as st

from app.scoring.rules import score_reading_test, ScoreResult
from app.ui.components import show_score_card, show_section_header


_LESETEXT = """
Die schnelle Entwicklung moderner Sehhilfen eröffnet neue Möglichkeiten
für präzise, individuelle Korrektionen. Beim Lesen dieses Textes achten
Sie bitte auf eventuelle Unschärfen, Verzerrungen oder Ermüdungserscheinungen.
Lesen Sie den gesamten Absatz ruhig und in Ihrem normalen Lesetempo durch.

Beobachten Sie dabei: Werden die Buchstaben am Rand unscharf? Müssen Sie
die Augen anstrengen? Empfinden Sie den Text als angenehm lesbar oder
eher unangenehm?
"""


def run_reading_test() -> ScoreResult | None:
    """Führt den Lese-Komfort-Test durch und gibt das Ergebnis zurück.

    Returns:
        ScoreResult wenn der Test abgeschlossen wurde, sonst None.
    """
    show_section_header("Lese-Komfort-Test", "")

    st.write("Lesen Sie bitte den folgenden Text und bewerten Sie danach Ihr Leseerlebnis.")

    # Lesetext anzeigen
    st.markdown(
        f"""
        <div style="
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            padding: 16px 20px;
            font-size: 1.05em;
            line-height: 1.7;
            color: #333;
            margin: 12px 0;
        ">
        {_LESETEXT.strip()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Bewerten Sie Ihr Leseerlebnis:**")

    anstrengung = st.slider(
        label="Wie anstrengend war das Lesen?",
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help="0 = keine Anstrengung  |  10 = sehr anstrengend",
    )

    unschaerfe = st.slider(
        label="Wie stark war die wahrgenommene Unschärfe?",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="0 = keine Unschärfe  |  10 = starke Unschärfe",
    )

    komfort = st.slider(
        label="Wie komfortabel war das Lesen insgesamt?",
        min_value=0,
        max_value=10,
        value=7,
        step=1,
        help="0 = sehr unbequem  |  10 = sehr komfortabel",
    )

    if st.button("Test auswerten", type="primary"):
        result = score_reading_test(
            anstrengung=anstrengung,
            unschaerfe=unschaerfe,
            komfort=komfort,
        )

        st.success("Auswertung abgeschlossen!")
        show_score_card(
            label=result["label"],
            score=result["score"],
            details=result["details"],
        )

        return result

    return None
