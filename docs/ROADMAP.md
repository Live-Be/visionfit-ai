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
- [x] Blink Detection (`app/cv/eye_metrics.py`)
  - EAR (Eye Aspect Ratio) nach Soukupova & Cech
  - `eye_aspect_ratio()`, `detect_blink()`, `blink_rate()`, `label_blink_rate()`
  - `summarize_eye_metrics()` für Multi-Frame-Sequenzen
  - `blink_rate_adjustment()` als optionaler Score-Faktor (scoring/rules.py)
  - `is_reliable=False` bei Einzelbild – vollständig vorbereitet für v0.3
- [x] Erweiterte Score-Formel mit Echtzeitdaten
  - Kombination: Bild-Score + Head Stability + Blink Rate
  - `score_fixation_combined()` (50/30/20-Gewichtung) in scoring/rules.py
- [x] Live-Kamera-Stream statt Einzelfoto (streamlit-webrtc, v0.3)

---

## v0.3 – Videoanalyse

**Ziel:** Zeitbasierte Analyse für präzisere Aussagen

- [x] Videoaufnahme statt Einzelfoto (3 Sekunden, ~90 Frames bei 30 fps)
  - streamlit-webrtc + FrameBuffer (Ring-Puffer, max. 150 Frames)
  - State Machine: idle → recording → analyzing → done
  - 3-Sekunden-Countdown mit Progress Bar + Frame-Counter
- [x] Frame-by-Frame-Analyse
  - Landmark-Extraktion für alle Frames (`app/cv/landmark_pipeline.py`)
  - Video-Modus (static_image_mode=False) für effizientes Tracking
  - face_detection_rate als Qualitäts- und Zuverlässigkeitsindikator
  - Fixationsstabilität als Zeitreihe via head_stability
- [x] Vollständige Analyse-Pipeline (`app/cv/video_analysis.py`)
  - Bildqualität aus mittlerem Frame (robuster als Frame 0)
  - Head Stability + Blink Detection orchestriert
  - is_reliable-Flag bei face_detection_rate < 0.3
- [x] Kombinierter Score (`score_fixation_combined`)
  - 50% Bildqualität + 30% Kopfstabilität + 20% Blinkmuster
  - Blink-Stressindikator: abnormale Rate ≤ 4 Punkte Abzug
- [x] pytest-Testabdeckung für v0.3 (`tests/test_video_analysis.py`, 60+ Tests)
- [x] Frame-Utilities (`app/cv/video_capture.py`)
  - validate_frames, build_frame_sequence, capture_frame_sequence (OpenCV)
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
