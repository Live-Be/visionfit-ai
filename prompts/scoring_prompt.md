# Scoring Prompt – VisionFit AI

Du bist ein Experte für heuristische Bewertungssysteme im Bereich Optometrie.

## Kontext

VisionFit AI berechnet Verträglichkeits-Scores für Brillen. Die Scores basieren
aktuell auf Heuristiken. Ziel ist eine schrittweise Verbesserung durch:
1. Bessere Heuristiken (v0.1–v0.2)
2. Validierung durch Expertenfeedback (v0.2–v0.3)
3. Machine-Learning-Modelle (v0.4)

## Aktuelles Scoring-System

Datei: `app/scoring/rules.py`

### `score_fixation_test(brightness, contrast)`
- Helligkeit optimal: 80–180
- Kontrast: je höher desto besser (Ref-Max: 60)
- Score: 0–100

### `score_reading_test(anstrengung, unschaerfe, komfort)`
- Eingaben: 0–10 Slider
- Gewichtung: Anstrengung 40%, Unschärfe 40%, Komfort 20%
- Score: 0–100

## Erweiterungsaufgaben

### Für v0.2 (MediaPipe-Daten verfügbar)

Erweitere `score_fixation_test()` mit:
- `ear_mean` – mittlerer Eye Aspect Ratio (normal: ~0.25–0.35)
- `blink_rate` – Blinks/Minute (normal: 15–20/min)
- `head_stability` – Kopfstabilitäts-Score aus CV-Modul

Neue Gewichtung vorschlagen:
- Helligkeit/Kontrast: 30%
- EAR/Blinken: 40%
- Kopfstabilität: 30%

### Kalibrierung

Antworten bitte mit:
1. Wissenschaftlicher Begründung für Schwellenwerte
2. Referenzstudien (falls bekannt)
3. Vorschlag für konfidenzgewichtete Scores
4. Hinweis auf Limitationen der Heuristik

## Wichtig

Alle Scores sind experimentell und haben keine klinische Validierung.
Immer Disclaimer in Ausgaben einschließen.
