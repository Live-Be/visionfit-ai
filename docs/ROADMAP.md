# VisionFit AI – Produktroadmap

---

## v0.1 – MVP (aktuell)

**Ziel:** Lauffähiger Prototyp mit grundlegenden Tests

- [x] Browserbasierte Streamlit-App
- [x] Fixationsstabilitäts-Test (Kamera + Bildanalyse)
- [x] Lese-Komfort-Test (Slider-Selbsteinschätzung)
- [x] Heuristischer Score (0–100) mit deutschem Label
- [x] Session-Speicherung als JSON
- [x] Modularer Python-Code
- [x] pytest-Testabdeckung für Scoring
- [x] Vollständig deutscher UI-Text
- [x] Disclaimer: kein Medizinprodukt

---

## v0.2 – MediaPipe Integration

**Ziel:** Objektive Gesichts- und Blickverfolgung

- [x] MediaPipe Face Mesh Integration (`app/cv/face_mesh.py`)
  - 478 Gesichts-Landmarks inkl. Iris erkennen
  - Gesichtsstatus und annotiertes Bild im Fixationstest
  - `get_landmark_xy()` als Basis für Phase 2/3
- [x] Head Stability Analyse (`app/cv/head_stability.py`)
  - Nasenspitzen-Tracking (Landmark 1) als Referenzpunkt
  - `std_x`, `std_y`, `combined_motion_std` aus Frame-Sequenz
  - Stabilitäts-Score (0–100) + deutsche Labels
  - `score_fixation_with_stability()` in scoring/rules.py (70/30-Gewichtung)
  - Vorbereitet für Multi-Frame-Video (v0.3)
- [ ] Blink Detection
  - Eye Aspect Ratio (EAR) berechnen
  - Blinkhäufigkeit und -dauer messen
  - Anomale Blinkraten als Stresssignal werten
- [ ] Erweiterte Score-Formel mit Echtzeitdaten
- [ ] Live-Kamera-Stream statt Einzelfoto

---

## v0.3 – Videoanalyse

**Ziel:** Zeitbasierte Analyse für präzisere Aussagen

- [ ] Videoaufnahme statt Einzelfoto (5–15 Sekunden)
- [ ] Frame-by-Frame-Analyse
  - Pupillenposition über Zeit
  - Microsaccaden erkennen
  - Fixationsstabilität als Zeitreihe
- [ ] Kontrastsensitivität via Grating-Test
  - Animierte Streifenmuster
  - Nutzerantwort (sichtbar / nicht sichtbar)
  - Schwellenwert-Bestimmung
- [ ] Glare-Empfindlichkeits-Test
  - Helligkeitsanpassung
  - Blendempfindlichkeits-Score
- [ ] Detaillierter PDF-Report

---

## v0.4 – Machine Learning

**Ziel:** Datengetriebene Verträglichkeitsprognose

- [ ] Trainingsdaten-Sammlung via Session-JSONs
- [ ] Feature Engineering aus CV-Metriken
  - EAR, Kopfstabilität, Blinkhäufigkeit
  - Selbsteinschätzungs-Slider
- [ ] Random Forest / Gradient Boosting Modell
  - Klassifikation: verträglich / nicht verträglich
  - Regression: Verträglichkeitsscore 0–100
- [ ] Modell-Serving via Streamlit + joblib
- [ ] Explainability: SHAP-Werte anzeigen
- [ ] A/B-Test: Heuristik vs. ML-Modell

---

## Langfristig (v1.0+)

- Cloud-Deployment (Streamlit Cloud / AWS)
- Benutzeranmeldung & Profilverwaltung
- Optiker-Dashboard mit Patientenübersicht
- DSGVO-konforme Datenspeicherung
- Klinische Validierungsstudie
- Integration mit Brillen-Produktkatalog
- API-Schnittstelle für Optikerverwaltungssoftware
