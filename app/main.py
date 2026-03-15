"""VisionFit AI – Haupt-Streamlit-Applikation."""

import streamlit as st

from app.ui.components import show_disclaimer
from app.tests.fixation_test import run_fixation_test
from app.tests.reading_test import run_reading_test
from app.storage.session_store import save_session
from app.utils.session import new_session_id, build_session_meta, current_timestamp
from app.utils.config import APP_NAME

# ──────────────────────────────────────────────
# Seiten-Konfiguration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="",
    layout="centered",
    initial_sidebar_state="auto",
)

# ──────────────────────────────────────────────
# Session-ID (einmalig pro Browser-Session)
# ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = new_session_id()

if "results" not in st.session_state:
    st.session_state["results"] = {}

SESSION_ID: str = st.session_state["session_id"]

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title(f" {APP_NAME}")
st.caption(f"Session-ID: `{SESSION_ID[:8]}…`  |  v0.1.0")

show_disclaimer()

st.markdown("---")

# ──────────────────────────────────────────────
# Testauswahl
# ──────────────────────────────────────────────
st.subheader("Test auswählen")

test_option = st.radio(
    label="Welchen Test möchten Sie durchführen?",
    options=["Fixationsstabilität", "Lese-Komfort"],
    horizontal=True,
)

st.markdown("---")

# ──────────────────────────────────────────────
# Test ausführen
# ──────────────────────────────────────────────
result = None

if test_option == "Fixationsstabilität":
    result = run_fixation_test()
    if result is not None:
        st.session_state["results"]["fixation"] = result

elif test_option == "Lese-Komfort":
    result = run_reading_test()
    if result is not None:
        st.session_state["results"]["reading"] = result

# ──────────────────────────────────────────────
# Session speichern
# ──────────────────────────────────────────────
if st.session_state["results"]:
    st.markdown("---")
    st.subheader(" Session speichern")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            f"Sie haben **{len(st.session_state['results'])}** Test(s) durchgeführt. "
            "Möchten Sie die Ergebnisse speichern?"
        )

    with col2:
        if st.button(" Ergebnisse speichern", type="primary"):
            session_data = {
                **build_session_meta(SESSION_ID),
                "saved_at": current_timestamp(),
                "results": st.session_state["results"],
            }
            file_path = save_session(session_data)
            st.success(f"Session gespeichert:")
            st.code(file_path, language="bash")

# ──────────────────────────────────────────────
# Alle bisherigen Ergebnisse
# ──────────────────────────────────────────────
if len(st.session_state["results"]) > 1:
    st.markdown("---")
    st.subheader(" Gesamtübersicht")

    total = sum(r["score"] for r in st.session_state["results"].values())
    avg = total / len(st.session_state["results"])

    st.metric(
        label="Durchschnittlicher Score",
        value=f"{avg:.0f} / 100",
        delta=None,
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    " Kein Medizinprodukt. Nur für Forschungs- und UX-Zwecke.  |  "
    "VisionFit AI © 2025"
)
