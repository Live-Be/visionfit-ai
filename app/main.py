"""VisionFit AI – Haupt-Streamlit-Applikation (v0.4 Optiker-Workflow)."""

import streamlit as st

from app.ui.components import show_disclaimer, show_score_card
from app.ui.forms import (
    render_anamnese_form,
    render_versorgung_form,
    render_binocular_form,
    get_refraction_values,
)
from app.tests.fixation_test import run_fixation_test
from app.tests.reading_test import run_reading_test
from app.storage.session_store import save_session
from app.utils.session import new_session_id, build_session_meta, current_timestamp
from app.utils.config import APP_NAME
from app.utils.refraction_compare import calculate_delta
from app.utils.design_ranking import rank_designs

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
# Session-Initialisierung
# ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state["session_id"] = new_session_id()

if "results" not in st.session_state:
    st.session_state["results"] = {}

SESSION_ID: str = st.session_state["session_id"]


# ──────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────

def _get_vsi_score():
    results = st.session_state.get("results", {})
    if "fixation" in results:
        return results["fixation"].get("score")
    if "reading" in results:
        return results["reading"].get("score")
    return None


def _render_vsi_summary(vsi_score):
    if vsi_score is None:
        st.info("Noch kein VisionFit Test durchgeführt (Tab 5).")
        return
    results = st.session_state.get("results", {})
    label_map = {"fixation": "Fixationsstabilität", "reading": "Lese-Komfort"}
    for key, result in results.items():
        show_score_card(
            label=label_map.get(key, key),
            score=result["score"],
            details=result.get("details", {}),
        )


def _render_binocular_context():
    phorie = st.session_state.get("bino_phorie", "nein")
    visus_r = st.session_state.get("bino_visus_r") or None
    visus_l = st.session_state.get("bino_visus_l") or None
    if phorie == "nein" and not visus_r and not visus_l:
        return
    st.markdown("**Binokularer Kontext**")
    cols = st.columns(3)
    if phorie != "nein":
        cols[0].metric("Phorie", phorie)
        cols[1].metric("Richtung", st.session_state.get("bino_phorie_richtung", "–"))
        cols[2].metric("Ausprägung", st.session_state.get("bino_phorie_auspraegung", "–"))
    if visus_r:
        st.caption(f"Visus R: {visus_r:.2f}  |  Visus L: {visus_l:.2f if visus_l else '–'}")


def _compute_ranking(vsi_score, delta):
    glastyp_neu = st.session_state.get("vers_neu_glastyp", "")
    if not glastyp_neu:
        return []
    return rank_designs(
        glastyp_neu=glastyp_neu,
        hauptanwendung=st.session_state.get("anam_hauptanwendung") or [],
        anlass=st.session_state.get("anam_anlass", ""),
        glastyp_alt=st.session_state.get("vers_alt_glastyp", ""),
        vertraeglichkeit_alt=st.session_state.get("vers_alt_vertraeglichkeit", ""),
        design_neu=st.session_state.get("vers_neu_design", ""),
        vsi_score=vsi_score,
        phorie=st.session_state.get("bino_phorie", "nein"),
        phorie_auspraegung=st.session_state.get("bino_phorie_auspraegung", ""),
        visus_r=st.session_state.get("bino_visus_r") or None,
        visus_l=st.session_state.get("bino_visus_l") or None,
        change_magnitude=delta["change_magnitude"] if delta else 0.0,
        beschwerden=st.session_state.get("anam_beschwerden") or [],
    )


def _render_ranking_item(item: dict) -> None:
    score = item["score"]
    if score >= 80:
        color = "#27ae60"
    elif score >= 60:
        color = "#2980b9"
    elif score >= 40:
        color = "#e67e22"
    else:
        color = "#c0392b"

    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            padding: 10px 14px;
            margin-bottom: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        ">
            <strong>{item['rang']}. {item['kategorie']}</strong>
            <span style="float:right; color:{color}; font-weight:bold;">{score}/100</span><br>
            <small style="color:#555;">{item['begruendung']}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_session_data(session_id: str) -> dict:
    delta = st.session_state.get("_delta")
    return {
        **build_session_meta(session_id),
        "saved_at": current_timestamp(),
        "anamnese": {
            "kundencode": st.session_state.get("anam_kundencode", ""),
            "alter": st.session_state.get("anam_alter", 0),
            "hauptanwendung": st.session_state.get("anam_hauptanwendung") or [],
            "anlass": st.session_state.get("anam_anlass", ""),
            "beschwerden": st.session_state.get("anam_beschwerden") or [],
        },
        "versorgung_alt": {
            "refraktion": get_refraction_values("vers_alt"),
            "glastyp": st.session_state.get("vers_alt_glastyp", ""),
            "design": st.session_state.get("vers_alt_design", ""),
            "vertraeglichkeit": st.session_state.get("vers_alt_vertraeglichkeit", ""),
        },
        "versorgung_neu": {
            "refraktion": get_refraction_values("vers_neu"),
            "glastyp": st.session_state.get("vers_neu_glastyp", ""),
            "design": st.session_state.get("vers_neu_design", ""),
            "bemerkung": st.session_state.get("vers_neu_bemerkung", ""),
        },
        "binokular": {
            "phorie": st.session_state.get("bino_phorie", "nein"),
            "phorie_richtung": st.session_state.get("bino_phorie_richtung", ""),
            "phorie_auspraegung": st.session_state.get("bino_phorie_auspraegung", ""),
            "visus_r": st.session_state.get("bino_visus_r", 0.0),
            "visus_l": st.session_state.get("bino_visus_l", 0.0),
            "visus_bino": st.session_state.get("bino_visus_bino", 0.0),
        },
        "delta_zusammenfassung": delta["summary"] if delta else "",
        "change_magnitude": delta["change_magnitude"] if delta else 0.0,
        "results": st.session_state.get("results", {}),
    }


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title(f" {APP_NAME}")
st.caption(f"Session-ID: `{SESSION_ID[:8]}…`  |  v0.4.0")

show_disclaimer()

# ──────────────────────────────────────────────
# Tab-Navigation
# ──────────────────────────────────────────────
tabs = st.tabs([
    "1 · Anamnese",
    "2 · Alte Versorgung",
    "3 · Neue Versorgung",
    "4 · Binokular",
    "5 · VisionFit Test",
    "6 · Ergebnis & Ranking",
])

# ── Tab 1: Anamnese ────────────────────────────
with tabs[0]:
    render_anamnese_form()

# ── Tab 2: Bisherige Versorgung ────────────────
with tabs[1]:
    render_versorgung_form("Bisherige Versorgung", "vers_alt")

# ── Tab 3: Neue Versorgung ─────────────────────
with tabs[2]:
    render_versorgung_form("Geplante neue Versorgung", "vers_neu")

# ── Tab 4: Binokular & Differenzanalyse ────────
with tabs[3]:
    render_binocular_form()

    st.divider()
    st.subheader("Differenzanalyse alt → neu")

    old_ref = get_refraction_values("vers_alt")
    new_ref = get_refraction_values("vers_neu")
    has_any = any(v is not None for k, v in {**old_ref, **new_ref}.items() if k != "prisma")

    if has_any:
        delta = calculate_delta(
            old_ref, new_ref,
            glastyp_alt=st.session_state.get("vers_alt_glastyp", ""),
            glastyp_neu=st.session_state.get("vers_neu_glastyp", ""),
            design_alt=st.session_state.get("vers_alt_design", ""),
            design_neu=st.session_state.get("vers_neu_design", ""),
        )
        st.info(delta["summary"])
        st.session_state["_delta"] = delta
    else:
        st.caption("Refraktionsdaten in Tab 2 & 3 eingeben, um die Differenzanalyse zu sehen.")
        st.session_state["_delta"] = None

# ── Tab 5: VisionFit Test ──────────────────────
with tabs[4]:
    st.subheader("VisionFit Test")
    st.markdown(
        "Kamerabasierter Fixationstest oder Lese-Komfort-Selbstauskunft."
    )

    test_option = st.radio(
        "Test auswählen:",
        options=["Fixationsstabilität (Kamera)", "Lese-Komfort (Selbstauskunft)"],
        horizontal=True,
        key="test_option_radio",
    )

    st.markdown("---")

    if test_option == "Fixationsstabilität (Kamera)":
        result = run_fixation_test()
        if result is not None:
            st.session_state["results"]["fixation"] = result

    elif test_option == "Lese-Komfort (Selbstauskunft)":
        result = run_reading_test()
        if result is not None:
            st.session_state["results"]["reading"] = result

    if st.session_state["results"]:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{len(st.session_state['results'])}** Test(s) abgeschlossen.")
        with col2:
            if st.button("Session speichern", type="primary", key="save_btn"):
                file_path = save_session(_build_session_data(SESSION_ID))
                st.success(f"Gespeichert: `{file_path}`")

# ── Tab 6: Ergebnis & Ranking ──────────────────
with tabs[5]:
    st.subheader("Ergebnisübersicht")

    vsi_score = _get_vsi_score()
    _render_vsi_summary(vsi_score)

    saved_delta = st.session_state.get("_delta")
    if saved_delta:
        st.markdown("**Optische Änderung**")
        st.info(saved_delta["summary"])

    _render_binocular_context()

    st.divider()
    st.subheader("Empfohlene Versorgungsstrategien")
    st.caption(
        "Heuristische, herstellerunabhängige Versorgungskategorien – kein medizinischer Anspruch."
    )

    ranking = _compute_ranking(vsi_score, saved_delta)
    if ranking:
        for item in ranking:
            _render_ranking_item(item)
    else:
        st.info(
            "Bitte in Tab 3 einen Glastyp auswählen, um Versorgungsempfehlungen zu erhalten."
        )

    if len(st.session_state["results"]) > 1:
        st.divider()
        total = sum(r["score"] for r in st.session_state["results"].values())
        avg = total / len(st.session_state["results"])
        st.metric("Durchschnittlicher Score", f"{avg:.0f} / 100")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption("Kein Medizinprodukt. Nur für Forschungs- und UX-Zwecke.  |  VisionFit AI © 2025")
