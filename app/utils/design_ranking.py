"""Herstellerunabhängiges, heuristisches Design-Ranking für Glasversorgungen.

Alle Empfehlungen sind rein heuristisch und ohne medizinischen Anspruch.
Es werden ausschließlich Design-/Versorgungskategorien genannt – keine Marken.
"""

from typing import Optional, TypedDict


class RankedDesign(TypedDict):
    rang: int
    kategorie: str
    score: int          # 0–100
    begruendung: str


# Interne Schlüssel → Anzeigename
_DESIGNS: dict[str, str] = {
    "standard_einstaerke": "Standard Einstärke",
    "asphaerische_einstaerke": "Asphärische Einstärke",
    "individualisierte_einstaerke": "Individualisierte Einstärke",
    "nahkomfort": "Nahkomfortglas",
    "office_raumglas": "Office / Raumglas",
    "bildschirmglas": "Bildschirmglas",
    "weiches_gleitsicht": "Weiches Gleitsichtdesign",
    "ausgewogenes_gleitsicht": "Ausgewogenes Gleitsichtdesign",
    "hartes_gleitsicht": "Hartes / Dynamisches Gleitsichtdesign",
    "individualisiertes_gleitsicht": "Individualisiertes Gleitsichtdesign",
    "binokular_sensibel": "Binokular sensible Versorgung",
    "prismatisch": "Prismatische Versorgung",
}

# Glastyp-neu → relevante Design-Gruppen (Keys)
_RELEVANT_BY_GLASTYP: dict[str, list[str]] = {
    "Einstärke": [
        "standard_einstaerke",
        "asphaerische_einstaerke",
        "individualisierte_einstaerke",
    ],
    "Gleitsicht": [
        "weiches_gleitsicht",
        "ausgewogenes_gleitsicht",
        "hartes_gleitsicht",
        "individualisiertes_gleitsicht",
    ],
    "Office": ["office_raumglas", "bildschirmglas", "nahkomfort"],
    "Nahkomfort": ["nahkomfort", "office_raumglas", "bildschirmglas"],
    "Arbeitsplatz": ["office_raumglas", "bildschirmglas", "nahkomfort"],
    "Bildschirmglas": ["bildschirmglas", "office_raumglas"],
    "Bifokal": ["weiches_gleitsicht", "ausgewogenes_gleitsicht"],
    "Spezialglas": list(_DESIGNS.keys()),
}

# Sonder-Kategorien, die immer geprüft werden
_SONDER = ["binokular_sensibel", "prismatisch"]


def rank_designs(
    glastyp_neu: str = "",
    hauptanwendung: Optional[list[str]] = None,
    anlass: str = "",
    glastyp_alt: str = "",
    vertraeglichkeit_alt: str = "",
    design_neu: str = "",
    vsi_score: Optional[float] = None,
    phorie: str = "nein",
    phorie_auspraegung: str = "",
    visus_r: Optional[float] = None,
    visus_l: Optional[float] = None,
    change_magnitude: float = 0.0,
    beschwerden: Optional[list[str]] = None,
) -> list[RankedDesign]:
    """Gibt eine heuristische Rangliste von Versorgungskategorien zurück.

    Alle Scores sind Schätzwerte ohne klinische Validierung.
    """
    if not glastyp_neu:
        return []

    hauptanwendung = hauptanwendung or []
    beschwerden = beschwerden or []

    scores: dict[str, int] = {k: 50 for k in _DESIGNS}

    _apply_glastyp_filter(scores, glastyp_neu)
    _apply_hauptanwendung_rules(scores, hauptanwendung)
    _apply_anlass_rules(scores, anlass)
    _apply_vertraeglichkeit_rules(scores, vertraeglichkeit_alt, beschwerden)
    _apply_vsi_rules(scores, vsi_score, change_magnitude)
    _apply_design_neu_rules(scores, design_neu)
    _apply_phorie_rules(scores, phorie, phorie_auspraegung)
    _apply_visus_rules(scores, visus_r, visus_l)
    _apply_wechsel_rules(scores, glastyp_alt, glastyp_neu)

    # Clamp to 0–100
    for k in scores:
        scores[k] = max(0, min(100, scores[k]))

    # Filter: nur Scores > 25 anzeigen
    visible = [(k, v) for k, v in scores.items() if v > 25]
    visible.sort(key=lambda x: x[1], reverse=True)
    visible = visible[:6]  # max. 6 Empfehlungen

    result: list[RankedDesign] = []
    for rang, (key, score) in enumerate(visible, start=1):
        result.append(
            RankedDesign(
                rang=rang,
                kategorie=_DESIGNS[key],
                score=score,
                begruendung=_build_explanation(
                    key, score, hauptanwendung, anlass, phorie,
                    vsi_score, vertraeglichkeit_alt, change_magnitude,
                    glastyp_alt, glastyp_neu,
                ),
            )
        )

    return result


# ── Regelmodule ────────────────────────────────────────────────────────────────

def _apply_glastyp_filter(scores: dict, glastyp_neu: str) -> None:
    """Nicht-relevante Kategorien stark abwerten."""
    relevant = set(_RELEVANT_BY_GLASTYP.get(glastyp_neu, list(_DESIGNS.keys())))
    relevant.update(_SONDER)
    for k in scores:
        if k not in relevant:
            scores[k] -= 40


def _apply_hauptanwendung_rules(scores: dict, anwendung: list[str]) -> None:
    if "Bildschirm" in anwendung:
        scores["bildschirmglas"] += 30
        scores["office_raumglas"] += 20
    if "Lesen" in anwendung:
        scores["nahkomfort"] += 25
    if "Arbeitsplatz" in anwendung or "Alltagsbrille" in anwendung:
        scores["office_raumglas"] += 20
    if "Autofahren" in anwendung or "Ferne" in anwendung:
        scores["hartes_gleitsicht"] += 10
        scores["asphaerische_einstaerke"] += 10
    if "Sport" in anwendung:
        scores["hartes_gleitsicht"] += 5
        scores["asphaerische_einstaerke"] += 10
    if "Gleitsicht" in anwendung:
        scores["ausgewogenes_gleitsicht"] += 15


def _apply_anlass_rules(scores: dict, anlass: str) -> None:
    if anlass == "Umgewöhnung":
        scores["weiches_gleitsicht"] += 15
        scores["ausgewogenes_gleitsicht"] += 5
        scores["hartes_gleitsicht"] -= 10
    elif anlass == "Unverträglichkeit":
        scores["weiches_gleitsicht"] += 20
        scores["hartes_gleitsicht"] -= 15
        scores["individualisiertes_gleitsicht"] += 10
    elif anlass == "Neuanpassung":
        scores["ausgewogenes_gleitsicht"] += 5
    elif anlass == "Vergleich mehrerer Glasoptionen":
        scores["individualisiertes_gleitsicht"] += 10
        scores["individualisierte_einstaerke"] += 10


def _apply_vertraeglichkeit_rules(
    scores: dict, vertraeglichkeit: str, beschwerden: list[str]
) -> None:
    if vertraeglichkeit == "schlecht":
        scores["weiches_gleitsicht"] += 20
        scores["hartes_gleitsicht"] -= 20
        scores["binokular_sensibel"] += 10
    elif vertraeglichkeit == "gut":
        scores["hartes_gleitsicht"] += 10
        scores["ausgewogenes_gleitsicht"] += 5
    elif vertraeglichkeit == "teilweise":
        scores["weiches_gleitsicht"] += 10
        scores["ausgewogenes_gleitsicht"] += 5

    if "Schwindel" in beschwerden or "peripheres Unwohlsein" in beschwerden:
        scores["weiches_gleitsicht"] += 15
        scores["hartes_gleitsicht"] -= 15
    if "Kopfschmerzen" in beschwerden:
        scores["binokular_sensibel"] += 10
        scores["weiches_gleitsicht"] += 10


def _apply_vsi_rules(
    scores: dict, vsi: Optional[float], magnitude: float
) -> None:
    if vsi is None:
        return
    if vsi < 40:
        scores["weiches_gleitsicht"] += 20
        scores["hartes_gleitsicht"] -= 20
        scores["ausgewogenes_gleitsicht"] += 10
    elif vsi < 65:
        scores["weiches_gleitsicht"] += 10
        scores["hartes_gleitsicht"] -= 5
    else:
        scores["hartes_gleitsicht"] += 10
        scores["ausgewogenes_gleitsicht"] += 5

    # Große Änderung + niedriger VSI → Extra-Vorsicht
    if magnitude > 0.5 and vsi < 60:
        scores["weiches_gleitsicht"] += 10
        scores["hartes_gleitsicht"] -= 10


def _apply_design_neu_rules(scores: dict, design_neu: str) -> None:
    if design_neu == "Individual":
        scores["individualisiertes_gleitsicht"] += 20
        scores["individualisierte_einstaerke"] += 15
    elif design_neu == "weich":
        scores["weiches_gleitsicht"] += 15
    elif design_neu == "hart":
        scores["hartes_gleitsicht"] += 15
    elif design_neu == "asphärisch":
        scores["asphaerische_einstaerke"] += 10


def _apply_phorie_rules(
    scores: dict, phorie: str, auspraegung: str
) -> None:
    if phorie == "ja":
        scores["binokular_sensibel"] += 35
        if auspraegung in ("mittel", "stark"):
            scores["prismatisch"] += 30
            scores["hartes_gleitsicht"] -= 10
        else:
            scores["prismatisch"] += 10
    elif phorie == "unklar":
        scores["binokular_sensibel"] += 15


def _apply_visus_rules(
    scores: dict,
    visus_r: Optional[float],
    visus_l: Optional[float],
) -> None:
    low_threshold = 0.6
    for v in [visus_r, visus_l]:
        if v is not None and v < low_threshold:
            # Niedrigvisus: einfachere, verlässlichere Designs bevorzugen
            scores["asphaerische_einstaerke"] += 5
            scores["individualisiertes_gleitsicht"] += 5
            scores["hartes_gleitsicht"] -= 5


def _apply_wechsel_rules(
    scores: dict, glastyp_alt: str, glastyp_neu: str
) -> None:
    """Wechsel zwischen Glastypen beeinflusst Adaptationslast."""
    wechsel = (glastyp_alt, glastyp_neu)
    if wechsel == ("Einstärke", "Gleitsicht") or wechsel == ("unbekannt", "Gleitsicht"):
        scores["weiches_gleitsicht"] += 15
        scores["ausgewogenes_gleitsicht"] += 5
    elif wechsel in [
        ("Gleitsicht", "Gleitsicht"),
        ("Individual", "Gleitsicht"),
    ]:
        scores["ausgewogenes_gleitsicht"] += 5
        scores["hartes_gleitsicht"] += 5


# ── Erklärungstexte ─────────────────────────────────────────────────────────

_ERKLAERUNGEN: dict[str, str] = {
    "standard_einstaerke": "Bewährte Grundversorgung für einfache sphärische oder zylindrische Korrekturen.",
    "asphaerische_einstaerke": "Reduzierte Bildfehler am Randbereich, dünneres Glas – gute Wahl für moderate bis höhere Stärken.",
    "individualisierte_einstaerke": "Auf individuelle Parameter angepasste Fertigung – empfehlenswert bei höheren Anforderungen.",
    "nahkomfort": "Erweiterte Nahentfernung mit leichter Progression – geringer Adaptationsaufwand bei Lesebedarf.",
    "office_raumglas": "Optimiert für Zwischenentfernungen (0,5–3 m) – besonders bei Bildschirm- und Büroumgebungen.",
    "bildschirmglas": "Auf digitale Bildschirmdistanzen (50–80 cm) abgestimmt – reduziert Anstrengung bei intensiver Bildschirmarbeit.",
    "weiches_gleitsicht": "Breite Progressionszone, geringere periphere Verzerrung – erleichtert Erstanpassung und Umgewöhnung.",
    "ausgewogenes_gleitsicht": "Ausgewogenes Verhältnis zwischen Fern-, Zwischen- und Nahbereich – vielseitig einsetzbar.",
    "hartes_gleitsicht": "Großer Fernbereich und klare Sehzonen – höchste Leistung bei guter Adaptationsfähigkeit.",
    "individualisiertes_gleitsicht": "Individuell berechnetes Design auf Basis persönlicher Parameter – geringste Abbildungsfehler.",
    "binokular_sensibel": "Versorgungsansatz mit besonderem Augenmerk auf binokulare Balance – relevant bei Phorie-Befund.",
    "prismatisch": "Prismatische Korrekturen zur binokularen Entlastung – indiziert bei relevantem Phoriewert.",
}


def _build_explanation(
    key: str,
    score: int,
    hauptanwendung: list[str],
    anlass: str,
    phorie: str,
    vsi: Optional[float],
    vertraeglichkeit: str,
    magnitude: float,
    glastyp_alt: str,
    glastyp_neu: str,
) -> str:
    base = _ERKLAERUNGEN.get(key, "")
    extras: list[str] = []

    if phorie == "ja" and key in ("binokular_sensibel", "prismatisch"):
        extras.append("Phorie-Befund vorhanden.")
    if vsi is not None and vsi < 50 and key == "weiches_gleitsicht":
        extras.append(f"VSI {vsi:.0f}/100 deutet auf erhöhte Adaptationsanforderung hin.")
    if "Bildschirm" in hauptanwendung and key in ("bildschirmglas", "office_raumglas"):
        extras.append("Hauptanwendung Bildschirm priorisiert diese Kategorie.")
    if anlass == "Unverträglichkeit" and key == "weiches_gleitsicht":
        extras.append("Anlass Unverträglichkeit – geringere Adaptationslast bevorzugt.")
    if glastyp_alt and glastyp_neu and glastyp_alt != glastyp_neu:
        extras.append(f"Versorgungswechsel {glastyp_alt} → {glastyp_neu} berücksichtigt.")
    if magnitude > 0.5:
        extras.append("Größere Refraktionsänderung – schrittweise Anpassung empfohlen.")

    hint = " ".join(extras)
    return f"{base} {hint}".strip()
