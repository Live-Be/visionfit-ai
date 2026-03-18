"""Refraktions-Vergleich: Delta-Berechnung zwischen alter und neuer Versorgung."""

from typing import Optional, TypedDict


class RefractionValues(TypedDict, total=False):
    sph_r: Optional[float]
    cyl_r: Optional[float]
    achse_r: Optional[float]
    sph_l: Optional[float]
    cyl_l: Optional[float]
    achse_l: Optional[float]
    add: Optional[float]
    prisma: Optional[str]


class RefractionDelta(TypedDict):
    delta_sph_r: Optional[float]
    delta_cyl_r: Optional[float]
    delta_achse_r: Optional[float]
    delta_sph_l: Optional[float]
    delta_cyl_l: Optional[float]
    delta_achse_l: Optional[float]
    delta_add: Optional[float]
    glastyp_wechsel: Optional[str]
    design_wechsel: Optional[str]
    summary: str
    change_magnitude: float  # 0.0–1.0: relative Gesamtgröße der Änderung


def _delta_float(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None:
        return None
    return round(new - old, 2)


def _achse_delta(old: Optional[float], new: Optional[float]) -> Optional[float]:
    """Achse-Delta: kürzester Weg auf dem 180°-Kreis."""
    if old is None or new is None:
        return None
    diff = new - old
    # Wrap to [-90, 90]
    while diff > 90:
        diff -= 180
    while diff < -90:
        diff += 180
    return round(diff, 1)


def calculate_delta(
    old: RefractionValues,
    new: RefractionValues,
    glastyp_alt: str = "",
    glastyp_neu: str = "",
    design_alt: str = "",
    design_neu: str = "",
) -> RefractionDelta:
    """Berechnet Deltas zwischen alter und neuer Versorgung."""
    d_sph_r = _delta_float(old.get("sph_r"), new.get("sph_r"))
    d_cyl_r = _delta_float(old.get("cyl_r"), new.get("cyl_r"))
    d_ach_r = _achse_delta(old.get("achse_r"), new.get("achse_r"))
    d_sph_l = _delta_float(old.get("sph_l"), new.get("sph_l"))
    d_cyl_l = _delta_float(old.get("cyl_l"), new.get("cyl_l"))
    d_ach_l = _achse_delta(old.get("achse_l"), new.get("achse_l"))
    d_add = _delta_float(old.get("add"), new.get("add"))

    glastyp_wechsel = (
        f"{glastyp_alt} → {glastyp_neu}"
        if glastyp_alt and glastyp_neu and glastyp_alt != glastyp_neu
        else None
    )
    design_wechsel = (
        f"{design_alt} → {design_neu}"
        if design_alt and design_neu and design_alt != design_neu
        else None
    )

    summary = _build_summary(
        d_sph_r, d_cyl_r, d_ach_r,
        d_sph_l, d_cyl_l, d_ach_l,
        d_add, glastyp_wechsel, design_wechsel,
    )

    magnitude = _change_magnitude(d_sph_r, d_cyl_r, d_sph_l, d_cyl_l, d_add)

    return RefractionDelta(
        delta_sph_r=d_sph_r,
        delta_cyl_r=d_cyl_r,
        delta_achse_r=d_ach_r,
        delta_sph_l=d_sph_l,
        delta_cyl_l=d_cyl_l,
        delta_achse_l=d_ach_l,
        delta_add=d_add,
        glastyp_wechsel=glastyp_wechsel,
        design_wechsel=design_wechsel,
        summary=summary,
        change_magnitude=magnitude,
    )


def _fmt(label: str, value: Optional[float], unit: str = "") -> str:
    if value is None:
        return ""
    sign = "+" if value > 0 else ""
    return f"{label}: {sign}{value}{unit}"


def _build_summary(
    d_sph_r, d_cyl_r, d_ach_r,
    d_sph_l, d_cyl_l, d_ach_l,
    d_add, glastyp_wechsel, design_wechsel,
) -> str:
    parts = []

    r_parts = [
        p for p in [
            _fmt("Sph", d_sph_r, " dpt"),
            _fmt("Cyl", d_cyl_r, " dpt"),
            _fmt("Achse", d_ach_r, "°"),
        ] if p
    ]
    if r_parts:
        parts.append("R: " + ", ".join(r_parts))

    l_parts = [
        p for p in [
            _fmt("Sph", d_sph_l, " dpt"),
            _fmt("Cyl", d_cyl_l, " dpt"),
            _fmt("Achse", d_ach_l, "°"),
        ] if p
    ]
    if l_parts:
        parts.append("L: " + ", ".join(l_parts))

    if d_add is not None:
        parts.append(_fmt("Add", d_add, " dpt"))

    if glastyp_wechsel:
        parts.append(f"Wechsel: {glastyp_wechsel}")

    if design_wechsel:
        parts.append(f"Design: {design_wechsel}")

    return " | ".join(parts) if parts else "Keine auswertbaren Refraktionsdaten."


def _change_magnitude(
    d_sph_r, d_cyl_r, d_sph_l, d_cyl_l, d_add
) -> float:
    """Schätzt Gesamtgröße der Änderung als Wert 0.0–1.0."""
    total = 0.0
    ref_max = 4.0  # ±4 dpt als Referenz für "große" Änderung
    for v in [d_sph_r, d_cyl_r, d_sph_l, d_cyl_l, d_add]:
        if v is not None:
            total += abs(v)
    return min(1.0, total / ref_max)
