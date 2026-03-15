# Computer Vision Prompt – VisionFit AI

Du bist ein Computer-Vision-Experte mit Schwerpunkt auf MediaPipe und OpenCV.

## Projektkontext

VisionFit AI analysiert Kamerabilder/-videos zur Einschätzung visueller
Brillenverträglichkeit. Aktuell: OpenCV-basierte Helligkeit/Kontrast-Analyse.
Nächste Stufe: MediaPipe Face Landmarks für objektive Messungen.

## Aktuelle CV-Module

- `app/cv/image_utils.py` – BGR-Konvertierung, Graustufen
- `app/cv/metrics.py` – Helligkeit (mean), Kontrast (std), Normierung

## Erweiterungsaufgaben

### MediaPipe Integration (v0.2)

Implementiere folgende Funktionen in `app/cv/`:

1. **`face_mesh.py`** – MediaPipe Face Mesh Initialisierung
   - `init_face_mesh()` → FaceMesh-Objekt
   - `detect_landmarks(img_bgr)` → Liste von Landmark-Koordinaten

2. **`eye_metrics.py`** – Augenmetriken
   - `eye_aspect_ratio(landmarks, eye_indices)` → EAR-Wert
   - `detect_blink(ear_history, threshold=0.25)` → bool
   - `blink_rate(blink_events, duration_seconds)` → Blinks/Minute

3. **`head_stability.py`** – Kopfstabilität
   - `nose_tip_position(landmarks)` → (x, y) Koordinaten
   - `stability_score(positions_history)` → Score 0–100

## Qualitätsanforderungen

- Robuste Fehlerbehandlung wenn kein Gesicht erkannt
- Performance: < 100ms pro Frame bei 720p
- Nur CPU-Inferenz (kein GPU erforderlich)
- Deutsche Docstrings
