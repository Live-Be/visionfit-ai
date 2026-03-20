"""Sakkadentest – Kamerabasierter Blicksprung-Test (VisionFit AI).

Misst horizontale Augensakkaden: Der Nutzer folgt einem springenden Punkt
abwechselnd links und rechts mit den Augen, während die Kamera die
Iris-Bewegungen aufzeichnet.

Berechnete Metriken:
    - Blicksprung-Reaktionszeit (saccade_latency_ms)
    - Zielgenauigkeit (accuracy_score)
    - Korrekturbewegungen (correction_saccades_count)
    - Links/Rechts-Symmetrie (symmetry_score)

UI-Flow (State Machine):
    idle       → Anleitung + Start-Button
    running    → Live-Kamera + springender Stimulus-Punkt
    analyzing  → Analyse läuft (Spinner)
    done       → Ergebnisse werden angezeigt

WICHTIGE HINWEISE:
    - Kein Medizinprodukt, keine klinische Diagnose.
    - Ergebnisse sind heuristische Näherungen.
    - Iris-Tracking benötigt gute Beleuchtung und frontale Kameraposition.
    - Bitte bei Sehproblemen einen Augenarzt oder Optiker konsultieren.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque

import numpy as np
import streamlit as st

from app.cv.saccade_analysis import analyze_saccade_test
from app.scoring.rules import ScoreResult, score_saccade_test
from app.ui.components import show_score_card, show_section_header

# ──────────────────────────────────────────────────────────────────────────────
# Optionaler Import: streamlit-webrtc
# ──────────────────────────────────────────────────────────────────────────────

try:
    import av  # noqa: F401
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # Dummy-Basisklasse

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

_N_TRANSITIONS: int = 12          # Anzahl Zielwechsel
_HOLD_TIME_MIN: float = 1.5       # Mindest-Haltezeit pro Position (Sekunden)
_HOLD_TIME_MAX: float = 2.8       # Maximal-Haltezeit pro Position (Sekunden)
_TARGET_FPS: float = 30.0

# Session-State-Schlüssel (isoliert, kein Konflikt mit anderen Tests)
_KEY_STATE = "saccade_v1_state"
_KEY_SEQUENCE = "saccade_v1_sequence"
_KEY_FRAMES = "saccade_v1_frames"
_KEY_EVENTS = "saccade_v1_events"
_KEY_RESULT = "saccade_v1_result"

_RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ──────────────────────────────────────────────────────────────────────────────
# Frame-Puffer mit Timestamps
# ──────────────────────────────────────────────────────────────────────────────

class SaccadeFrameBuffer(VideoProcessorBase):
    """Sammelt Video-Frames mit absoluten Timestamps thread-sicher.

    Die recv()-Methode wird von streamlit-webrtc in einem separaten Thread
    aufgerufen. Frames werden als (frame, timestamp) Tupel gespeichert.
    """

    def __init__(self) -> None:
        self._buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=600)
        self._lock = threading.Lock()

    def recv(self, frame):
        """Empfängt Frame + Timestamp, gibt Frame unverändert zurück."""
        img = frame.to_ndarray(format="bgr24")
        t = time.time()
        with self._lock:
            self._buffer.append((img, t))
        return frame

    def get_timed_frames(self) -> list[tuple[np.ndarray, float]]:
        """Gibt eine Kopie aller (frame, timestamp) Paare zurück."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Leert den Puffer."""
        with self._lock:
            self._buffer.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Stimulus-Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def _generate_stimulus_sequence(
    n_transitions: int = _N_TRANSITIONS,
) -> list[dict]:
    """Erzeugt eine randomisierte Stimulus-Sequenz.

    Richtung alterniert strikt links/rechts. Haltezeiten sind uniform
    zufällig im Bereich [_HOLD_TIME_MIN, _HOLD_TIME_MAX], um antizipatorische
    Augenbewegungen zu erschweren.

    Args:
        n_transitions: Anzahl Zielwechsel (MVP: 12).

    Returns:
        Liste von {'direction': 'left'|'right', 'hold_time': float}.
    """
    rng = random.Random()  # Nicht-deterministisch für jeden Testdurchlauf
    direction = rng.choice(["left", "right"])
    sequence = []
    for _ in range(n_transitions):
        sequence.append({
            "direction": direction,
            "hold_time": rng.uniform(_HOLD_TIME_MIN, _HOLD_TIME_MAX),
        })
        direction = "right" if direction == "left" else "left"
    return sequence


def _render_stimulus(
    placeholder,
    direction: str,
    step: int,
    total: int,
) -> None:
    """Rendert das Stimulus-Bild (Punkt links oder rechts).

    Zeigt zwei Kreise: der aktive (Ziel) in Blau, der inaktive in Grau.
    Beide Positionen sind bei 20% und 80% der Breite platziert –
    sicher innerhalb des sichtbaren Bereichs.

    Args:
        placeholder: Streamlit-Platzhalter (st.empty()).
        direction:   'left' oder 'right'.
        step:        Aktueller Schritt (0-basiert).
        total:       Gesamtzahl der Schritte.
    """
    left_color = "#2980b9" if direction == "left" else "#cccccc"
    right_color = "#2980b9" if direction == "right" else "#cccccc"

    placeholder.markdown(
        f"""
        <div style="
            position: relative;
            height: 120px;
            background: #f5f5f5;
            border-radius: 10px;
            margin: 8px 0;
            border: 1px solid #ddd;
        ">
            <div style="
                position: absolute;
                left: 20%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 44px;
                height: 44px;
                background: {left_color};
                border-radius: 50%;
                transition: background 0.1s;
            "></div>
            <div style="
                position: absolute;
                left: 80%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 44px;
                height: 44px;
                background: {right_color};
                border-radius: 50%;
                transition: background 0.1s;
            "></div>
            <div style="
                position: absolute;
                bottom: 8px;
                left: 50%;
                transform: translateX(-50%);
                color: #888;
                font-size: 0.75em;
            ">
                Schritt {step + 1} / {total}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ergebnisdarstellung
# ──────────────────────────────────────────────────────────────────────────────

def _render_saccade_results(analysis: dict, score_result: ScoreResult) -> None:
    """Zeigt die Sakkadentest-Ergebnisse strukturiert an."""
    st.markdown("---")
    st.markdown("#### Analyseergebnis")

    # Metriken-Übersicht in Spalten
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latency = analysis.get("latency_ms_mean")
        st.metric(
            "Reaktionszeit",
            f"{latency:.0f} ms" if latency is not None else "–",
            help="Mittlere Zeit zwischen Zielwechsel und Augenbewegung",
        )

    with col2:
        acc = analysis.get("accuracy_score", 0.0)
        st.metric(
            "Zielgenauigkeit",
            f"{acc:.0f} %",
            help="Anteil der Zielwechsel mit erkannter Sakkade",
        )

    with col3:
        corr = analysis.get("correction_saccades_count", 0)
        st.metric(
            "Korrekturbewegungen",
            str(corr),
            help="Nachkorrektur-Sakkaden nach initialem Landing",
        )

    with col4:
        sym = analysis.get("symmetry_score")
        st.metric(
            "Links/Rechts",
            f"{sym:.0f} / 100" if sym is not None else "–",
            help="Symmetrie zwischen linker und rechter Sakkade (100 = perfekt)",
        )

    # Interpretation
    interpretation = analysis.get("interpretation_text", "")
    if interpretation:
        st.info(f"**Muster:** {interpretation}")

    # Warnungen
    if analysis.get("head_movement_warning"):
        st.warning(
            "Kopfbewegung erkannt: Die Ergebnisse könnten durch Kopfbewegungen "
            "beeinflusst sein. Bei Wiederholung bitte den Kopf ruhig halten."
        )

    if not analysis.get("is_reliable"):
        st.warning(
            f"Eingeschränkte Datenqualität (Score: {analysis.get('quality_score', 0):.0f}/100). "
            f"Nur {analysis.get('analyzed_event_count', 0)} von "
            f"{analysis.get('raw_event_count', 0)} Sakkaden auswertbar. "
            "Für bessere Ergebnisse: gute Beleuchtung, Gesicht mittig im Bild, "
            "Abstand 40–70 cm."
        )

    # Score-Karte
    show_score_card(
        label=score_result["label"],
        score=score_result["score"],
        details=score_result["details"],
    )

    # Technische Details
    with st.expander("Technische Qualitätsinfos"):
        st.write(f"**Gesicht erkannt:** {analysis.get('face_detection_rate', 0) * 100:.0f}%")
        st.write(f"**Iris erkannt:** {analysis.get('iris_detection_rate', 0) * 100:.0f}%")
        st.write(
            f"**Auswertbare Sakkaden:** "
            f"{analysis.get('analyzed_event_count', 0)} / {analysis.get('raw_event_count', 0)}"
        )

    st.markdown(
        "_Hinweis: Alle Werte sind heuristische Näherungen. "
        "Kein Medizinprodukt, keine Diagnose._"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Haupt-Funktion
# ──────────────────────────────────────────────────────────────────────────────

def run_saccade_test() -> ScoreResult | None:
    """Führt den kamerabasierten Sakkadentest durch.

    Steuert den kompletten UI-Flow via State Machine.
    Gibt das ScoreResult zurück sobald die Analyse abgeschlossen ist,
    sonst None.

    Returns:
        ScoreResult nach abgeschlossener Analyse, sonst None.
    """
    show_section_header("Sakkadentest – Blicksprung-Messung", "")

    # ── Fallback wenn streamlit-webrtc nicht installiert ──────────────────────
    if not _WEBRTC_AVAILABLE:
        st.error(
            "**streamlit-webrtc nicht installiert.**  \n"
            "Bitte installieren Sie die erforderlichen Pakete:  \n"
            "```\npip install streamlit-webrtc av\n```  \n"
            "Anschließend die App neu starten."
        )
        return None

    # ── State initialisieren ──────────────────────────────────────────────────
    if _KEY_STATE not in st.session_state:
        st.session_state[_KEY_STATE] = "idle"

    state: str = st.session_state[_KEY_STATE]

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: idle – Anleitung + Start-Button
    # ────────────────────────────────────────────────────────────────────────
    if state == "idle":
        st.info(
            "**Anleitung:**  \n"
            "1. Setzen Sie sich ruhig vor die Kamera (Abstand: ca. 40–70 cm).  \n"
            "2. Folgen Sie dem springenden **blauen Punkt** nur mit den Augen.  \n"
            "3. Halten Sie den Kopf möglichst still.  \n"
            "4. Der Test dauert ca. 25–35 Sekunden."
        )

        webrtc_ctx = webrtc_streamer(
            key="saccade-v1-idle",
            video_processor_factory=SaccadeFrameBuffer,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=_RTC_CONFIG,
        )

        camera_active = webrtc_ctx.state.playing if hasattr(webrtc_ctx, "state") else False
        if not camera_active:
            st.caption("Kamera aktivieren und dann auf **Test starten** klicken.")
            return None

        col_hint, col_btn = st.columns([3, 1])
        with col_hint:
            st.markdown(
                "Kamera aktiv. Positionieren Sie Ihr Gesicht mittig im Bild.  \n"
                "Beide Augen und die Nase sollten gut sichtbar sein."
            )
        with col_btn:
            if st.button("Test starten", type="primary", use_container_width=True):
                sequence = _generate_stimulus_sequence(_N_TRANSITIONS)
                st.session_state[_KEY_SEQUENCE] = sequence
                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.clear()
                st.session_state[_KEY_STATE] = "running"
                st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: running – Live-Stimulus + Frame-Aufnahme
    # ────────────────────────────────────────────────────────────────────────
    elif state == "running":
        sequence: list[dict] = st.session_state.get(_KEY_SEQUENCE, [])
        if not sequence:
            st.error("Stimulus-Sequenz fehlt. Bitte neu starten.")
            st.session_state[_KEY_STATE] = "idle"
            st.rerun()
            return None

        webrtc_ctx = webrtc_streamer(
            key="saccade-v1-running",
            video_processor_factory=SaccadeFrameBuffer,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=_RTC_CONFIG,
        )

        stimulus_placeholder = st.empty()
        status_placeholder = st.empty()
        frame_counter = st.empty()

        # Stimulus-Schleife – blockierend, ~20 Hz UI-Updates
        stimulus_events: list[dict] = []
        current_step = 0
        step_start = time.time()

        # Erstes Event sofort registrieren
        stimulus_events.append({
            "time": step_start,
            "direction": sequence[current_step]["direction"],
        })

        while current_step < len(sequence):
            elapsed_in_step = time.time() - step_start
            step = sequence[current_step]

            _render_stimulus(stimulus_placeholder, step["direction"], current_step, len(sequence))

            n_frames = 0
            if webrtc_ctx.video_processor:
                n_frames = len(webrtc_ctx.video_processor.get_timed_frames())
            frame_counter.caption(f"Frames aufgenommen: {n_frames}")

            remaining_total = sum(
                s["hold_time"] for s in sequence[current_step:]
            ) - elapsed_in_step
            status_placeholder.caption(f"Verbleibend: ca. {max(0, remaining_total):.0f}s")

            if elapsed_in_step >= step["hold_time"]:
                current_step += 1
                if current_step < len(sequence):
                    step_start = time.time()
                    stimulus_events.append({
                        "time": step_start,
                        "direction": sequence[current_step]["direction"],
                    })
                continue

            time.sleep(0.05)  # ~20 Hz

        # Test abgeschlossen – Frames holen
        stimulus_placeholder.empty()
        status_placeholder.empty()
        frame_counter.empty()

        timed_frames: list[tuple[np.ndarray, float]] = []
        if webrtc_ctx.video_processor:
            timed_frames = webrtc_ctx.video_processor.get_timed_frames()

        st.session_state[_KEY_FRAMES] = timed_frames
        st.session_state[_KEY_EVENTS] = stimulus_events
        st.session_state[_KEY_STATE] = "analyzing"
        st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: analyzing
    # ────────────────────────────────────────────────────────────────────────
    elif state == "analyzing":
        with st.spinner("Sakkadenanalyse läuft… bitte warten."):
            timed_frames = st.session_state.get(_KEY_FRAMES, [])
            stimulus_events = st.session_state.get(_KEY_EVENTS, [])

            if not timed_frames:
                st.warning(
                    "Keine Frames aufgenommen. Bitte Kamera-Zugriff prüfen "
                    "und den Test erneut starten."
                )
                st.session_state[_KEY_STATE] = "idle"
                st.rerun()
                return None

            analysis = analyze_saccade_test(timed_frames, stimulus_events)

            score_result = score_saccade_test(
                accuracy_score=analysis["accuracy_score"],
                latency_ms_mean=analysis["latency_ms_mean"],
                correction_saccades_count=analysis["correction_saccades_count"],
                quality_score=analysis["quality_score"],
                is_reliable=analysis["is_reliable"],
            )

            st.session_state[_KEY_RESULT] = {
                "analysis": analysis,
                "score_result": score_result,
            }
            st.session_state[_KEY_STATE] = "done"
            st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: done – Ergebnisse anzeigen
    # ────────────────────────────────────────────────────────────────────────
    elif state == "done":
        stored = st.session_state.get(_KEY_RESULT)
        if stored is None:
            st.session_state[_KEY_STATE] = "idle"
            st.rerun()
            return None

        analysis = stored["analysis"]
        score_result = stored["score_result"]

        # Fehlermeldung bei ungültiger Analyse
        if analysis.get("warnung"):
            st.error(f"**Test nicht auswertbar:** {analysis['warnung']}")
            if st.button("Neu starten", key="saccade_retry_btn"):
                _reset_state()
                st.rerun()
            return None

        _render_saccade_results(analysis, score_result)

        if st.button("Neuen Test starten", key="saccade_new_btn"):
            _reset_state()
            st.rerun()

        return score_result

    return None


def _reset_state() -> None:
    """Setzt alle Sakkaden-Test-Session-Keys zurück."""
    for key in (_KEY_STATE, _KEY_SEQUENCE, _KEY_FRAMES, _KEY_EVENTS, _KEY_RESULT):
        st.session_state.pop(key, None)
