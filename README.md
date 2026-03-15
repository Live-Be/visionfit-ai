# VisionFit AI

**Kamerabasierter Prototyp zur Einschätzung visueller Verträglichkeit von Brillen**

> **Hinweis:** VisionFit AI ist **kein Medizinprodukt** und **kein Diagnosetool**.
> Dieser Prototyp dient ausschließlich Forschungs- und UX-Zwecken.
> Bei gesundheitlichen Fragen konsultieren Sie bitte einen Augenarzt oder Optiker.

---

## Projektbeschreibung

VisionFit AI ist ein deutschsprachiger Web-Prototyp auf Basis von Streamlit.
Ziel ist es, die visuelle Verträglichkeit verschiedener Brillen durch kamerabasierte
Tests und Selbsteinschätzungen heuristisch zu bewerten.

---

## Aktueller Stand – v0.3 (Video MVP)

### Fixationsstabilitäts-Test (Video-basiert)

- **3-Sekunden-Videoaufnahme** via Browser-Kamera (`streamlit-webrtc`)
- **Face Landmark Detection** – 478 Gesichtspunkte via MediaPipe FaceMesh
- **Head Stability** – Nasenspitzen-Tracking über alle Frames; Score 0–100
- **Blink Detection** – Eye Aspect Ratio (EAR) nach Soukupova & Cech; Blinkrate/Minute
- **Kombinierter Score** – 50 % Bildqualität + 30 % Kopfstabilität + 20 % Blinkmuster
- **Stressindikator** – abnormale Blinkrate reduziert Score leicht (max. ~4 Punkte)
- **Zuverlässigkeitsprüfung** – `face_detection_rate < 30 %` → ungültiger Test

### Lese-Komfort-Test

- Selbsteinschätzung via Slider (Anstrengung, Unschärfe, Komfort)
- Heuristischer Score 0–100

### Allgemein

- Vollständig deutsche UI-Texte
- Session-Speicherung als JSON
- 271 pytest-Tests grün
- Streamlit-App läuft lokal im Browser (Desktop + mobil testbar)

---

## Projektstruktur

```
app/
├── main.py                    # Streamlit-Einstiegspunkt
├── cv/
│   ├── face_mesh.py           # MediaPipe Face Landmarks
│   ├── head_stability.py      # Kopfstabilitäts-Analyse
│   ├── eye_metrics.py         # EAR-basierte Blink Detection
│   ├── video_capture.py       # Frame-Utilities (OpenCV)
│   ├── landmark_pipeline.py   # Landmark-Extraktion aus Frame-Sequenz
│   ├── video_analysis.py      # Vollständige Analyse-Pipeline
│   ├── image_utils.py         # Bildumwandlungen
│   └── metrics.py             # Bildmetriken
├── scoring/
│   └── rules.py               # Scoring-Heuristiken
├── tests/
│   ├── fixation_test.py       # Fixationstest-Screen (v0.3 Video)
│   └── reading_test.py        # Lesetest-Screen
├── ui/
│   └── components.py          # Wiederverwendbare UI-Komponenten
├── storage/
│   └── session_store.py       # JSON-Session-Speicherung
└── utils/
    ├── config.py              # Konfiguration via .env
    └── session.py             # Session-ID-Generierung

tests/                         # pytest-Tests
docs/ROADMAP.md                # Produktroadmap
data/sessions/                 # Gespeicherte Sessions (JSON)
```

---

## Setup & Start

### Voraussetzungen

- Python 3.10+
- Browser mit Kamerazugriff (Chrome / Firefox empfohlen)

### Installation

```bash
gh repo clone Live-Be/visionfit-ai
cd visionfit-ai

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### App starten

```bash
cd ~/dev/visionfit-ai
source .venv/bin/activate
PYTHONPATH=$PWD streamlit run app/main.py
```

Die App öffnet sich unter: **http://localhost:8501**

> **PYTHONPATH=$PWD** ist erforderlich damit alle `app.*`-Imports korrekt aufgelöst werden.

### Für den Fixationstest (Video) wird benötigt:

```bash
pip install streamlit-webrtc av
```

Ohne diese Pakete startet die App, der Fixationstest zeigt jedoch eine
Installations-Hinweismeldung statt der Kameraaufnahme.

---

## Tests ausführen

```bash
cd ~/dev/visionfit-ai
source .venv/bin/activate
PYTHONPATH=$PWD python -m pytest tests/ -v
```

Aktuell: **271 Tests, alle grün.**

---

## Browser / Mobile

Die App ist im lokalen Netzwerk über die IP des Entwicklungsrechners erreichbar:

```bash
PYTHONPATH=$PWD streamlit run app/main.py --server.address 0.0.0.0
```

Dann im Browser (Handy): `http://<DEINE-IP>:8501`

`streamlit-webrtc` benötigt für den Videozugriff auf dem Handy **HTTPS** oder
`localhost`. Für einen echten mobilen Test empfiehlt sich ein HTTPS-Tunnel
(z.B. `ngrok http 8501`).

---

## Umgebungsvariablen

Kopieren Sie `.env.example` nach `.env`:

| Variable      | Standard         | Beschreibung                  |
|---------------|-----------------|-------------------------------|
| `APP_ENV`     | `development`    | Umgebung (development/prod)   |
| `APP_NAME`    | `VisionFit AI`   | App-Name                      |
| `SESSION_DIR` | `data/sessions`  | Speicherort für Session-JSONs |

---

## Rechtlicher Hinweis

VisionFit AI ist **kein Medizinprodukt** gemäß MDR (EU) 2017/745 und erfüllt keine
regulatorischen Anforderungen an Diagnosewerkzeuge für medizinische Zwecke.
Alle angezeigten Scores sind experimentelle Heuristiken ohne klinische Validierung.
