"""Tests für app.utils.design_ranking."""

import pytest
from app.utils.design_ranking import rank_designs, RankedDesign


class TestRankDesigns:
    def test_returns_list(self):
        result = rank_designs(glastyp_neu="Gleitsicht")
        assert isinstance(result, list)

    def test_result_structure(self):
        result = rank_designs(glastyp_neu="Gleitsicht")
        for item in result:
            assert "rang" in item
            assert "kategorie" in item
            assert "score" in item
            assert "begruendung" in item

    def test_rang_sequential(self):
        result = rank_designs(glastyp_neu="Gleitsicht")
        for i, item in enumerate(result, start=1):
            assert item["rang"] == i

    def test_scores_clamped(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            phorie="ja",
            phorie_auspraegung="stark",
            vsi_score=10.0,
            vertraeglichkeit_alt="schlecht",
            anlass="Unverträglichkeit",
        )
        for item in result:
            assert 0 <= item["score"] <= 100

    def test_empty_glastyp_returns_empty(self):
        result = rank_designs(glastyp_neu="")
        assert result == []

    def test_bildschirm_prioritizes_office(self):
        result = rank_designs(
            glastyp_neu="Office",
            hauptanwendung=["Bildschirm"],
        )
        kategorien = [r["kategorie"] for r in result]
        assert len(kategorien) > 0
        # Office oder Bildschirmglas sollte vorne sein
        assert any("Office" in k or "Bildschirm" in k for k in kategorien[:2])

    def test_phorie_triggers_binokular(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            phorie="ja",
            phorie_auspraegung="mittel",
        )
        kategorien = [r["kategorie"] for r in result]
        assert "Binokular sensible Versorgung" in kategorien

    def test_strong_phorie_triggers_prismatisch(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            phorie="ja",
            phorie_auspraegung="stark",
        )
        kategorien = [r["kategorie"] for r in result]
        assert "Prismatische Versorgung" in kategorien

    def test_low_vsi_prefers_weich(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            vsi_score=30.0,
        )
        kategorien = [r["kategorie"] for r in result]
        # Weiches Gleitsichtdesign sollte in Top-3 sein
        assert "Weiches Gleitsichtdesign" in kategorien[:3]

    def test_high_vsi_allows_hartes(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            vsi_score=90.0,
            vertraeglichkeit_alt="gut",
        )
        kategorien = [r["kategorie"] for r in result]
        # Hartes Gleitsicht sollte erscheinen (nicht gefiltert)
        assert any("Hart" in k or "Dynamisch" in k for k in kategorien)

    def test_einstaerke_shows_einstaerke_designs(self):
        result = rank_designs(glastyp_neu="Einstärke")
        kategorien = [r["kategorie"] for r in result]
        assert any("Einstärke" in k for k in kategorien)

    def test_schlechte_vertraeglichkeit_penalizes_hartes(self):
        result_schlecht = rank_designs(
            glastyp_neu="Gleitsicht",
            vertraeglichkeit_alt="schlecht",
        )
        result_gut = rank_designs(
            glastyp_neu="Gleitsicht",
            vertraeglichkeit_alt="gut",
        )

        def score_for(results, kategorie):
            for r in results:
                if r["kategorie"] == kategorie:
                    return r["score"]
            return None

        hart_schlecht = score_for(result_schlecht, "Hartes / Dynamisches Gleitsichtdesign")
        hart_gut = score_for(result_gut, "Hartes / Dynamisches Gleitsichtdesign")

        if hart_schlecht is not None and hart_gut is not None:
            assert hart_schlecht < hart_gut

    def test_unvertraeglichkeit_anlass_prefers_weich(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            anlass="Unverträglichkeit",
        )
        kategorien = [r["kategorie"] for r in result]
        assert "Weiches Gleitsichtdesign" in kategorien[:2]

    def test_max_6_results(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            phorie="ja",
            phorie_auspraegung="stark",
            hauptanwendung=["Bildschirm", "Lesen"],
        )
        assert len(result) <= 6

    def test_begruendung_not_empty(self):
        result = rank_designs(glastyp_neu="Gleitsicht")
        for item in result:
            assert len(item["begruendung"]) > 0

    def test_wechsel_einstaerke_zu_gleitsicht_prefers_weich(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            glastyp_alt="Einstärke",
        )
        kategorien = [r["kategorie"] for r in result]
        assert "Weiches Gleitsichtdesign" in kategorien[:3]

    def test_change_magnitude_large_raises_weich(self):
        result_large = rank_designs(
            glastyp_neu="Gleitsicht",
            change_magnitude=0.8,
            vsi_score=55.0,
        )
        result_small = rank_designs(
            glastyp_neu="Gleitsicht",
            change_magnitude=0.1,
            vsi_score=55.0,
        )

        def weich_score(r):
            for item in r:
                if item["kategorie"] == "Weiches Gleitsichtdesign":
                    return item["score"]
            return 0

        assert weich_score(result_large) >= weich_score(result_small)

    def test_individuell_design_raises_individualisiert(self):
        result = rank_designs(
            glastyp_neu="Gleitsicht",
            design_neu="Individual",
        )
        for item in result:
            if item["kategorie"] == "Individualisiertes Gleitsichtdesign":
                assert item["score"] > 50
                break
