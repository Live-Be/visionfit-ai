"""Wiederverwendbare Streamlit-Formulare für den Optiker-Workflow (v0.4)."""

from typing import Optional
import streamlit as st


# ── Anamnese ──────────────────────────────────────────────────────────────────

def render_anamnese_form() -> None:
    """Rendert das Anamnese-Formular und speichert Werte in st.session_state."""
    st.subheader("Anamnese & Sehanforderung")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(
            "Kundencode / Test-ID (optional)",
            key="anam_kundencode",
            placeholder="z. B. KD-001",
        )
    with col2:
        st.number_input(
            "Alter (optional)",
            min_value=0,
            max_value=120,
            value=0,
            step=1,
            key="anam_alter",
            help="0 = keine Angabe",
        )

    st.multiselect(
        "Hauptanwendung",
        options=[
            "Ferne", "Lesen", "Bildschirm", "Alltagsbrille",
            "Gleitsicht", "Arbeitsplatz", "Autofahren", "Sport",
        ],
        key="anam_hauptanwendung",
    )

    st.selectbox(
        "Anlass",
        options=[
            "Neuanpassung", "Umgewöhnung", "Unverträglichkeit",
            "Vergleich mehrerer Glasoptionen", "Kontrolle",
        ],
        key="anam_anlass",
    )

    st.multiselect(
        "Bisherige Beschwerden",
        options=[
            "Unschärfe", "Schwindel", "Eingewöhnung schwierig",
            "Kopfschmerzen", "Lesen anstrengend",
            "peripheres Unwohlsein", "keine Beschwerden",
        ],
        key="anam_beschwerden",
    )


# ── Versorgungsformular (alt / neu) ───────────────────────────────────────────

def render_versorgung_form(label: str, key_prefix: str) -> None:
    """Rendert ein Versorgungsformular (alt oder neu).

    Args:
        label: Anzeige-Titel, z. B. 'Bisherige Versorgung'.
        key_prefix: Präfix für alle Session-State-Keys, z. B. 'vers_alt'.
    """
    st.subheader(label)

    st.markdown("**Refraktion**")
    col_r, col_l = st.columns(2)

    with col_r:
        st.markdown("*Rechts (R)*")
        st.number_input("Sph R", step=0.25, format="%.2f",
                        key=f"{key_prefix}_sph_r", value=0.0)
        st.number_input("Cyl R", step=0.25, format="%.2f",
                        key=f"{key_prefix}_cyl_r", value=0.0)
        st.number_input("Achse R (°)", step=1, min_value=0, max_value=180,
                        key=f"{key_prefix}_achse_r", value=0)

    with col_l:
        st.markdown("*Links (L)*")
        st.number_input("Sph L", step=0.25, format="%.2f",
                        key=f"{key_prefix}_sph_l", value=0.0)
        st.number_input("Cyl L", step=0.25, format="%.2f",
                        key=f"{key_prefix}_cyl_l", value=0.0)
        st.number_input("Achse L (°)", step=1, min_value=0, max_value=180,
                        key=f"{key_prefix}_achse_l", value=0)

    col_add, col_prisma = st.columns(2)
    with col_add:
        st.number_input("Addition (Add)", step=0.25, format="%.2f",
                        key=f"{key_prefix}_add", value=0.0)
    with col_prisma:
        st.text_input("Prisma (optional)",
                      key=f"{key_prefix}_prisma",
                      placeholder="z. B. 2Δ BI")

    st.divider()

    glastyp_options = _glastyp_options(key_prefix)
    st.selectbox("Glastyp", options=glastyp_options, key=f"{key_prefix}_glastyp")

    design_options = _design_options(key_prefix)
    st.selectbox("Design", options=design_options, key=f"{key_prefix}_design")

    if key_prefix.endswith("_alt"):
        st.selectbox(
            "Verträglichkeit bisherige Versorgung",
            options=["unbekannt", "gut", "teilweise", "schlecht"],
            key=f"{key_prefix}_vertraeglichkeit",
        )
    else:
        st.text_area(
            "Bemerkung zur Versorgung (optional)",
            key=f"{key_prefix}_bemerkung",
            height=80,
        )


def _glastyp_options(key_prefix: str) -> list[str]:
    if key_prefix.endswith("_alt"):
        return [
            "unbekannt", "Einstärke", "Gleitsicht", "Bifokal",
            "Office", "Nahkomfort", "Bildschirmglas", "Kontaktlinse",
        ]
    return [
        "Einstärke", "Gleitsicht", "Office", "Nahkomfort",
        "Arbeitsplatz", "Spezialglas",
    ]


def _design_options(key_prefix: str) -> list[str]:
    if key_prefix.endswith("_alt"):
        return [
            "unbekannt", "Standard", "Individual",
            "weich", "hart", "sphärisch", "asphärisch", "atorisch",
        ]
    return [
        "Standard", "Individual",
        "weich", "hart", "sphärisch", "asphärisch", "atorisch",
    ]


# ── Binokularbefunde ──────────────────────────────────────────────────────────

def render_binocular_form() -> None:
    """Rendert das Formular für binokulare Basisbefunde."""
    st.subheader("Binokularbefunde")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox(
            "Phorie vorhanden?",
            options=["nein", "ja", "unklar"],
            key="bino_phorie",
        )
    with col2:
        st.selectbox(
            "Phorierichtung",
            options=["unklar", "Exo", "Eso", "Hyper", "Vertikal"],
            key="bino_phorie_richtung",
        )
    with col3:
        st.selectbox(
            "Phorieausprägung",
            options=["unklar", "leicht", "mittel", "stark"],
            key="bino_phorie_auspraegung",
        )

    st.divider()
    st.markdown("**Erreichter Visus**")
    col_vr, col_vl, col_vb = st.columns(3)
    with col_vr:
        st.number_input(
            "Visus R", min_value=0.0, max_value=2.0,
            step=0.05, format="%.2f",
            key="bino_visus_r", value=0.0,
            help="0 = keine Angabe",
        )
    with col_vl:
        st.number_input(
            "Visus L", min_value=0.0, max_value=2.0,
            step=0.05, format="%.2f",
            key="bino_visus_l", value=0.0,
            help="0 = keine Angabe",
        )
    with col_vb:
        st.number_input(
            "Visus binokular", min_value=0.0, max_value=2.0,
            step=0.05, format="%.2f",
            key="bino_visus_bino", value=0.0,
            help="0 = keine Angabe",
        )


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def get_refraction_values(key_prefix: str) -> dict:
    """Liest Refraktionswerte aus session_state und gibt ein Dict zurück."""
    def _float_or_none(k: str) -> Optional[float]:
        v = st.session_state.get(k, 0.0)
        return float(v) if v not in (0.0, None) else None

    return {
        "sph_r": _float_or_none(f"{key_prefix}_sph_r"),
        "cyl_r": _float_or_none(f"{key_prefix}_cyl_r"),
        "achse_r": _float_or_none(f"{key_prefix}_achse_r"),
        "sph_l": _float_or_none(f"{key_prefix}_sph_l"),
        "cyl_l": _float_or_none(f"{key_prefix}_cyl_l"),
        "achse_l": _float_or_none(f"{key_prefix}_achse_l"),
        "add": _float_or_none(f"{key_prefix}_add"),
        "prisma": st.session_state.get(f"{key_prefix}_prisma") or None,
    }
