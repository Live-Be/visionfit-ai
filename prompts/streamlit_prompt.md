# Streamlit Prompt – VisionFit AI

Du bist ein Streamlit-Experte für mobile-optimierte Web-Apps.

## Projektkontext

VisionFit AI nutzt Streamlit als Frontend-Framework. Die App muss sowohl
auf Desktop-Browsern als auch auf mobilen Geräten gut funktionieren.

## Aktuelle UI-Struktur

- `app/main.py` – Page Config, Navigation, Session-Verwaltung
- `app/ui/components.py` – Wiederverwendbare Komponenten
- `app/tests/fixation_test.py` – Fixationstest-Screen
- `app/tests/reading_test.py` – Lesetest-Screen

## Aufgaben & Best Practices

### Mobile Optimierung

- `st.camera_input` für native Kamera-Integration
- Responsive Layout mit `st.columns()` sparsam einsetzen
- Touch-freundliche Button-Größen via CSS
- `layout="centered"` für Mobile-First

### State Management

- `st.session_state` für persistente Daten über Re-Runs
- Initialisierung immer am Anfang prüfen: `if "key" not in st.session_state`
- Session-ID bei App-Start einmalig generieren

### Performance

- `st.spinner()` für asynchrone Operationen anzeigen
- `st.cache_data` für teure Berechnungen cachen
- Bilder vor Analyse auf max. 800px Breite reduzieren

### UX-Richtlinien

- Klare Call-to-Actions (primäre Buttons)
- Disclaimer immer sichtbar (st.info)
- Fortschrittsanzeige bei mehrstufigen Tests
- Fehlermeldungen auf Deutsch mit Lösungsvorschlag

## Komponenten-Erweiterungen

Neue Komponenten in `app/ui/components.py` hinzufügen:
- `show_progress_bar(current, total)` – Testfortschritt
- `show_comparison_chart(results)` – Balkendiagramm der Scores
- `show_export_button(session_data)` – CSV/PDF-Export
