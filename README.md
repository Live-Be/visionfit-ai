# VisionFit AI

**Kamerabasierter Prototyp zur Einschätzung visueller Verträglichkeit von Brillen**

> **Hinweis:** VisionFit AI ist **kein Medizinprodukt** und **kein Diagnosetool**.
> Dieser Prototyp dient ausschließlich Forschungs- und UX-Zwecken.
> Bei gesundheitlichen Fragen konsultieren Sie bitte einen Augenarzt oder Optiker.

---

## Projektbeschreibung

VisionFit AI ist ein deutschsprachiger Web-Prototyp, der mit Streamlit umgesetzt wurde.
Ziel ist es, die visuelle Verträglichkeit verschiedener Brillen durch einfache, kamerabasierte
Tests und Selbsteinschätzungen heuristisch zu bewerten.

### Features (v0.1 MVP)

- Browserbasierte App – mobil und Desktop nutzbar
- Vollständig deutscher UI-Text
- **Fixationsstabilitäts-Test** – Kameraaufnahme + Bildanalyse (Helligkeit & Kontrast)
- **Lese-Komfort-Test** – Selbsteinschätzung via Slider
- Heuristischer Score (0–100) mit deutschem Label
- Session-Speicherung als JSON
- Modularer Python-Code mit klarer Struktur
- Vorbereitete Struktur für Computer-Vision-Erweiterungen

---

## Projektstruktur

```
app/
├── main.py                 # Streamlit-Einstiegspunkt
├── ui/
│   └── components.py       # Wiederverwendbare UI-Komponenten
├── cv/
│   ├── image_utils.py      # Bildumwandlungen
│   └── metrics.py          # Bildmetriken
├── tests/
│   ├── fixation_test.py    # Fixationstest-Screen
│   └── reading_test.py     # Lesetest-Screen
├── scoring/
│   └── rules.py            # Scoring-Heuristiken
├── storage/
│   └── session_store.py    # JSON-Speicherung
└── utils/
    ├── config.py            # Konfiguration via .env
    └── session.py           # Session-ID-Generierung

data/sessions/              # Gespeicherte Sessions (JSON)
docs/                       # Dokumentation
prompts/                    # KI-Prompt-Vorlagen
tests/
└── test_rules.py           # pytest-Tests
```

---

## Setup & Start

### Voraussetzungen

- Python 3.10+
- Git

### Installation

```bash
# 1. Repository klonen (falls noch nicht geschehen)
gh repo clone Live-Be/visionfit-ai
cd visionfit-ai

# 2. Virtuelle Umgebung erstellen und aktivieren
python3 -m venv .venv
source .venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Umgebungsvariablen setzen (optional)
cp .env.example .env
```

### App starten

```bash
# Variante A – direkt
streamlit run app/main.py

# Variante B – via Startskript
bash run.sh
```

Die App öffnet sich automatisch unter: **http://localhost:8501**

---

## Tests ausführen

```bash
pytest
```

Oder mit ausführlicher Ausgabe:

```bash
pytest -v
```

---

## Umgebungsvariablen

Kopieren Sie `.env.example` nach `.env` und passen Sie die Werte an:

| Variable      | Standard           | Beschreibung                  |
|---------------|-------------------|-------------------------------|
| `APP_ENV`     | `development`      | Umgebung (development/prod)   |
| `APP_NAME`    | `VisionFit AI`     | App-Name                      |
| `SESSION_DIR` | `data/sessions`    | Speicherort für Session-JSONs |

---

## Rechtlicher Hinweis

VisionFit AI ist **kein Medizinprodukt** gemäß MDR (EU) 2017/745 und erfüllt keine
regulatorischen Anforderungen an Diagnosewerkzeuge für medizinische Zwecke.
Alle angezeigten Scores sind experimentelle Heuristiken ohne klinische Validierung.
