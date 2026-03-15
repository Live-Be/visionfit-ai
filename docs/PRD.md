# Product Requirements Document – VisionFit AI

**Version:** 0.1.0
**Datum:** 2025
**Status:** MVP in Entwicklung

---

## 1. Produktidee

### Vision

VisionFit AI soll Menschen dabei unterstützen, die visuelle Verträglichkeit ihrer Brille
objektiv und reproduzierbar einzuschätzen. Durch den Einsatz von Computer Vision und
KI-gestützter Analyse soll der Prozess der Brillenanpassung verbessert werden.

### Problem

Die Beurteilung, ob eine Brille gut verträglich ist, basiert heute fast ausschließlich
auf subjektivem Feedback des Trägers und der Expertise des Optikers. Ein Prototyp, der
objektive Messwerte (Fixationsstabilität, Kontrastreiz-Reaktion, Blinkverhalten) erfasst,
kann diesen Prozess unterstützen.

### Zielgruppe

- **Primär:** Optiker und Augenoptik-Fachbetriebe als Screening-Tool
- **Sekundär:** Endverbraucher zur Selbsteinschätzung
- **Tertiär:** Forschungseinrichtungen im Bereich Optometrie

---

## 2. Scope MVP (v0.1)

### In Scope

- Browserbasierte Streamlit-App (mobil + Desktop)
- Fixationsstabilitäts-Test via Kameraaufnahme
- Lese-Komfort-Test via Selbsteinschätzung
- Heuristischer Score (0–100)
- Session-Speicherung als JSON
- Vollständig deutscher UI-Text

### Out of Scope (v0.1)

- Benutzeranmeldung / Authentifizierung
- Datenbankanbindung
- Cloud-Deployment
- Klinische Validierung
- DSGVO-konforme Datenspeicherung

---

## 3. Functional Requirements

| ID    | Anforderung                                    | Priorität |
|-------|------------------------------------------------|-----------|
| FR-01 | App läuft im Browser (Streamlit)               | Muss      |
| FR-02 | Kamera-Zugriff für Fixationstest               | Muss      |
| FR-03 | Slider-basierter Lese-Komfort-Test             | Muss      |
| FR-04 | Score-Ausgabe (0–100 + Label)                  | Muss      |
| FR-05 | Session-Speicherung als JSON                   | Muss      |
| FR-06 | Disclaimer (kein Medizinprodukt)               | Muss      |
| FR-07 | Session-ID zur Nachverfolgung                  | Soll      |
| FR-08 | Gesamtscore bei mehreren Tests                 | Soll      |
| FR-09 | Export-Funktion für Ergebnisse                 | Kann      |

---

## 4. Non-Functional Requirements

- **Performance:** Analyseergebnis < 2 Sekunden nach Foto-Upload
- **Kompatibilität:** Aktuelle Browser (Chrome, Safari, Firefox)
- **Sprache:** Vollständig Deutsch
- **Code-Qualität:** Modularer Python-Code, pytest-Abdeckung
- **Datenschutz:** Kein Upload von Bilddaten an externe Server

---

## 5. User Stories

### US-01 – Fixationstest
Als **Nutzer** möchte ich ein Kamerabild aufnehmen können, damit der Prototyp meine
Fixationsstabilität einschätzen kann.

**Akzeptanzkriterien:**
- Kamera-Input ist vorhanden
- Bild wird analysiert (Helligkeit, Kontrast)
- Score wird angezeigt

### US-02 – Lesetest
Als **Nutzer** möchte ich mein subjektives Leseerlebnis bewerten können, damit der
Prototyp einen Lese-Komfort-Score berechnet.

**Akzeptanzkriterien:**
- Lesetext wird angezeigt
- Slider für Anstrengung, Unschärfe, Komfort
- Score-Auswertung nach Klick

### US-03 – Session-Speicherung
Als **Nutzer** möchte ich meine Testergebnisse speichern können, damit ich sie später
mit meinem Optiker besprechen kann.

**Akzeptanzkriterien:**
- Speicher-Button vorhanden
- JSON-Datei wird erstellt
- Dateipfad wird angezeigt

---

## 6. Metriken & Erfolg

- App startet ohne Fehler: ✅
- Fixationstest liefert Score: ✅
- Lesetest liefert Score: ✅
- pytest-Tests bestehen: ✅
- Nutzerfeedback: Verständlichkeit > 7/10
