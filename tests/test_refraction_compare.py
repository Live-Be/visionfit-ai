"""Tests für app.utils.refraction_compare."""

import pytest
from app.utils.refraction_compare import (
    calculate_delta,
    RefractionValues,
)


def _ref(**kwargs) -> RefractionValues:
    defaults = dict(
        sph_r=None, cyl_r=None, achse_r=None,
        sph_l=None, cyl_l=None, achse_l=None,
        add=None, prisma=None,
    )
    defaults.update(kwargs)
    return RefractionValues(**defaults)


class TestCalculateDelta:
    def test_zero_delta(self):
        old = _ref(sph_r=1.0, sph_l=-0.5)
        new = _ref(sph_r=1.0, sph_l=-0.5)
        d = calculate_delta(old, new)
        assert d["delta_sph_r"] == 0.0
        assert d["delta_sph_l"] == 0.0

    def test_positive_sph_delta(self):
        old = _ref(sph_r=1.0)
        new = _ref(sph_r=1.5)
        d = calculate_delta(old, new)
        assert d["delta_sph_r"] == pytest.approx(0.5)

    def test_negative_cyl_delta(self):
        old = _ref(cyl_r=-0.5)
        new = _ref(cyl_r=-1.25)
        d = calculate_delta(old, new)
        assert d["delta_cyl_r"] == pytest.approx(-0.75)

    def test_add_delta(self):
        old = _ref(add=1.0)
        new = _ref(add=1.5)
        d = calculate_delta(old, new)
        assert d["delta_add"] == pytest.approx(0.5)

    def test_none_propagation(self):
        old = _ref(sph_r=None)
        new = _ref(sph_r=1.0)
        d = calculate_delta(old, new)
        assert d["delta_sph_r"] is None

    def test_both_none_propagation(self):
        old = _ref(sph_r=None)
        new = _ref(sph_r=None)
        d = calculate_delta(old, new)
        assert d["delta_sph_r"] is None

    def test_glastyp_wechsel_detected(self):
        old = _ref()
        new = _ref()
        d = calculate_delta(old, new, glastyp_alt="Einstärke", glastyp_neu="Gleitsicht")
        assert d["glastyp_wechsel"] == "Einstärke → Gleitsicht"

    def test_glastyp_same_no_wechsel(self):
        old = _ref()
        new = _ref()
        d = calculate_delta(old, new, glastyp_alt="Gleitsicht", glastyp_neu="Gleitsicht")
        assert d["glastyp_wechsel"] is None

    def test_design_wechsel(self):
        old = _ref()
        new = _ref()
        d = calculate_delta(old, new, design_alt="Standard", design_neu="Individual")
        assert d["design_wechsel"] == "Standard → Individual"

    def test_summary_contains_r_and_l(self):
        old = _ref(sph_r=1.0, sph_l=-1.0)
        new = _ref(sph_r=1.5, sph_l=-0.5)
        d = calculate_delta(old, new)
        assert "R:" in d["summary"]
        assert "L:" in d["summary"]

    def test_empty_summary_when_no_data(self):
        old = _ref()
        new = _ref()
        d = calculate_delta(old, new)
        assert "Keine auswertbaren" in d["summary"]

    def test_change_magnitude_zero_when_no_data(self):
        old = _ref()
        new = _ref()
        d = calculate_delta(old, new)
        assert d["change_magnitude"] == pytest.approx(0.0)

    def test_change_magnitude_large_change(self):
        old = _ref(sph_r=0.0, sph_l=0.0, cyl_r=0.0, cyl_l=0.0)
        new = _ref(sph_r=3.0, sph_l=3.0, cyl_r=-1.0, cyl_l=-1.0)
        d = calculate_delta(old, new)
        assert d["change_magnitude"] >= 0.5

    def test_change_magnitude_clamped_to_one(self):
        old = _ref(sph_r=0.0, sph_l=0.0)
        new = _ref(sph_r=10.0, sph_l=10.0)
        d = calculate_delta(old, new)
        assert d["change_magnitude"] == pytest.approx(1.0)

    def test_achse_delta_wraparound(self):
        """Achse-Delta soll den kürzesten Weg nehmen."""
        old = _ref(achse_r=5.0)
        new = _ref(achse_r=175.0)
        d = calculate_delta(old, new)
        # Kürzester Weg: -10° (nicht +170°)
        assert d["delta_achse_r"] == pytest.approx(-10.0)

    def test_achse_delta_direct(self):
        old = _ref(achse_r=30.0)
        new = _ref(achse_r=50.0)
        d = calculate_delta(old, new)
        assert d["delta_achse_r"] == pytest.approx(20.0)

    def test_summary_contains_glastyp_wechsel(self):
        old = _ref(sph_r=1.0)
        new = _ref(sph_r=1.5)
        d = calculate_delta(old, new, glastyp_alt="Einstärke", glastyp_neu="Gleitsicht")
        assert "Einstärke" in d["summary"]
        assert "Gleitsicht" in d["summary"]
