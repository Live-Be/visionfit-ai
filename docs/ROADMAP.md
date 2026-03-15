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

- [ ] MediaPipe Face Mesh Integration
  - 468 Gesichts-Landmarks erkennen
  - Augenbereich automatisch segmentieren
- [ ] Blink Detection
  - Eye Aspect Ratio (EAR) berechnen
  - Blinkhäufigkeit und -dauer messen
  - Anomale Blinkraten als Stresssignal werten
- [ ] Head Stability Analyse
  - Kopfpositions-Tracking via Landmarks
  - Bewegungsamplitude über Zeit berechnen
  - Stabilitäts-Score für Fixationstest
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
