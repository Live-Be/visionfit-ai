"""Fixationsstabilitäts-Test – Video-basiert mit MediaPipe Face Landmarks (v0.3).

Verwendet streamlit-webrtc für browserbasierte Live-Kameraaufnahme.
Frames werden 3 Sekunden lang gesammelt und anschließend vollständig
analysiert (Bildqualität, Kopfstabilität, Blinkrate, Kombinierter Score).

UI-Flow (State Machine):
    idle       → Kamera aktiv, Benutzer klickt „Test starten"
    recording  → 3-Sekunden-Countdown, Frames werden gesammelt
    analyzing  → Analyse läuft (Spinner)
    done       → Ergebnisse werden angezeigt

Fallback: Wenn streamlit-webrtc nicht installiert ist, erscheint eine
klare Fehlermeldung mit Installationsanweisung.

WICHTIGE HINWEISE:
    - Kein Medizinprodukt, keine klinische Diagnose.
    - Ergebnisse sind heuristische Näherungen.
    - Blinkrate und Stabilitätswerte abhängig von Kamera und Beleuchtung.
    - Bitte konsultieren Sie bei Sehproblemen einen Augenarzt oder Optiker.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import streamlit as st

from app.cv.video_capture import build_frame_sequence
from app.cv.video_analysis import analyze_video_sequence
from app.scoring.rules import (
    ScoreResult,
    score_fixation_combined,
    score_fixation_no_face,
)
from app.ui.components import show_score_card, show_section_header

# ──────────────────────────────────────────────────────────────────────────────
# Optionaler Import: streamlit-webrtc
# ──────────────────────────────────────────────────────────────────────────────

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # Dummy-Basisklasse für FrameBuffer-Definition

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

_RECORD_DURATION: float = 3.0          # Aufnahmedauer in Sekunden
_TARGET_FPS: float = 30.0              # Angestrebte FPS (kameraabhängig)
_MIN_FACE_RATE: float = 0.3            # Mindest-Gesichtserkennungsrate

# Session-State-Schlüssel (isoliert um Konflikte mit anderen Tests zu vermeiden)
_KEY_STATE = "fixation_v3_state"
_KEY_RECORD_START = "fixation_v3_record_start"
_KEY_FRAMES = "fixation_v3_frames"
_KEY_RESULT = "fixation_v3_result"

# WebRTC ICE-Server (STUN für lokale Entwicklung)
_RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ──────────────────────────────────────────────────────────────────────────────
# Frame-Puffer für streamlit-webrtc
# ──────────────────────────────────────────────────────────────────────────────

class FrameBuffer(VideoProcessorBase):
    """Sammelt Video-Frames thread-sicher in einem Ring-Puffer.

    Die recv()-Methode wird von streamlit-webrtc in einem separaten Thread
    aufgerufen. Zugriff auf den Puffer erfolgt via Lock.
    """

    def __init__(self) -> None:
        self._buffer: deque[np.ndarray] = deque(maxlen=150)
        self._lock = threading.Lock()

    def recv(self, frame):  # av.VideoFrame – kein Typ-Import im Fallback-Pfad
        """Empfängt einen Frame, speichert ihn im Puffer, gibt ihn unverändert zurück."""
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._buffer.append(img)
        return frame

    def get_frames(self) -> list[np.ndarray]:
        """Gibt eine Kopie des aktuellen Puffers zurück."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Leert den Puffer (vor neuer Aufnahme aufrufen)."""
        with self._lock:
            self._buffer.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Haupt-Funktion
# ──────────────────────────────────────────────────────────────────────────────

def run_fixation_test() -> ScoreResult | None:
    """Führt den Video-basierten Fixationstest durch.

    Steuert den kompletten UI-Flow via State Machine.
    Gibt das ScoreResult zurück sobald die Analyse abgeschlossen ist,
    sonst None.

    Returns:
        ScoreResult nach abgeschlossener Analyse, sonst None.
    """
    show_section_header("Fixationsstabilitäts-Test", "")

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

    # ── Instructions ─────────────────────────────────────────────────────────
    st.info(
        "**Bitte schauen Sie 3 Sekunden ruhig in die Kamera.**  \n"
        "Blinzeln Sie natürlich – erzwungenes Offenhalten der Augen verfälscht das Ergebnis.  \n"
        "Halten Sie Ihren Kopf möglichst still und sorgen Sie für gute Beleuchtung."
    )

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: idle oder recording – WebRTC-Stream anzeigen
    # ────────────────────────────────────────────────────────────────────────
    if state in ("idle", "recording"):
        webrtc_ctx = webrtc_streamer(
            key="fixation-v3",
            video_processor_factory=FrameBuffer,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=_RTC_CONFIG,
        )

        # Kamera noch nicht aktiv
        if not (webrtc_ctx.state.playing if hasattr(webrtc_ctx, "state") else False):
            st.caption("Kamera aktivieren und dann auf **Test starten** klicken.")
            return None

        # ── IDLE: Start-Button ────────────────────────────────────────────
        if state == "idle":
            col_hint, col_btn = st.columns([3, 1])
            with col_hint:
                st.markdown(
                    "Kamera aktiv. Positionieren Sie Ihr Gesicht mittig im Bild.  \n"
                    "Abstand zur Kamera: ca. 40–70 cm."
                )
            with col_btn:
                if st.button("Test starten", type="primary", use_container_width=True):
                    if webrtc_ctx.video_processor:
                        webrtc_ctx.video_processor.clear()
                    st.session_state[_KEY_STATE] = "recording"
                    st.session_state[_KEY_RECORD_START] = time.time()
                    st.rerun()

        # ── RECORDING: Countdown ─────────────────────────────────────────
        elif state == "recording":
            record_start = st.session_state.get(_KEY_RECORD_START, time.time())

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            frame_counter = st.empty()

            # Blocking-Countdown (3 Sekunden, 10 Hz Updates)
            while True:
                elapsed = time.time() - record_start
                remaining = max(0.0, _RECORD_DURATION - elapsed)

                progress_bar.progress(
                    min(elapsed / _RECORD_DURATION, 1.0),
                    text=f"Aufnahme läuft… {remaining:.1f}s verbleibend",
                )
                if webrtc_ctx.video_processor:
                    n = len(webrtc_ctx.video_processor.get_frames())
                    frame_counter.caption(f"Frames aufgenommen: {n}")

                if elapsed >= _RECORD_DURATION:
                    break
                time.sleep(0.1)

            progress_bar.progress(1.0, text="Aufnahme abgeschlossen.")

            # Frames aus Puffer holen
            frames: list[np.ndarray] = []
            if webrtc_ctx.video_processor:
                frames = webrtc_ctx.video_processor.get_frames()

            st.session_state[_KEY_FRAMES] = frames
            st.session_state[_KEY_STATE] = "analyzing"
            st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: analyzing
    # ────────────────────────────────────────────────────────────────────────
    elif state == "analyzing":
        with st.spinner("Analyse läuft… bitte warten."):
            raw_frames = st.session_state.get(_KEY_FRAMES, [])

            if not raw_frames:
                st.warning(
                    "Keine Frames aufgenommen. Bitte Kamera-Zugriff prüfen "
                    "und den Test erneut starten."
                )
                st.session_state[_KEY_STATE] = "idle"
                st.rerun()
                return None

            # Frame-Sequenz aufbereiten
            sequence = build_frame_sequence(raw_frames, fps=_TARGET_FPS)

            # Vollständige Video-Analyse
            analysis = analyze_video_sequence(
                sequence["frames"],
                fps=sequence["fps"],
            )

            # Score berechnen
            if analysis["face_detection_rate"] < _MIN_FACE_RATE:
                result = score_fixation_no_face()
            else:
                result = score_fixation_combined(
                    brightness=analysis["brightness"],
                    contrast=analysis["contrast"],
                    head_stability_score=analysis["head_stability_score"],
                    blink_rate=analysis["blink_rate"],
                )

            # Analyse-Details zum Ergebnis hinzufügen
            result["details"].update({
                "aufgenommene_frames": analysis["frame_count"],
                "gesicht_erkannt_%": f"{analysis['face_detection_rate'] * 100:.0f}%",
                "kopfstabilität": analysis["head_stability_label"],
                "blinkrate_pro_min": round(analysis["blink_rate"] or 0.0, 1),
                "ear_mittelwert": round(analysis["ear_mean"] or 0.0, 3),
            })

            st.session_state[_KEY_RESULT] = result
            st.session_state[_KEY_STATE] = "done"
            st.rerun()

    # ────────────────────────────────────────────────────────────────────────
    # Zustand: done – Ergebnisse anzeigen
    # ────────────────────────────────────────────────────────────────────────
    elif state == "done":
        result = st.session_state.get(_KEY_RESULT)
        if result is None:
            st.session_state[_KEY_STATE] = "idle"
            st.rerun()
            return None

        st.markdown("---")
        st.markdown("#### Analyseergebnis")

        # Kennzahlen-Übersicht
        col1, col2, col3 = st.columns(3)
        with col1:
            face_pct = result["details"].get("gesicht_erkannt_%", "–")
            st.metric("Gesicht erkannt", face_pct)
        with col2:
            stability = result["details"].get("kopfstabilität", "–")
            st.metric("Fixationsstabilität", stability)
        with col3:
            blink_r = result["details"].get("blinkrate_pro_min", "–")
            st.metric("Blinkrate / Min", str(blink_r))

        # Score-Karte
        show_score_card(
            label=result["label"],
            score=result["score"],
            details=result["details"],
        )

        st.markdown(
            "_Hinweis: Alle Werte sind heuristische Näherungen. "
            "Kein Medizinprodukt, keine Diagnose._"
        )

        # Reset-Button
        if st.button("Neuen Test starten"):
            for key in (_KEY_STATE, _KEY_FRAMES, _KEY_RESULT, _KEY_RECORD_START):
                st.session_state.pop(key, None)
            st.rerun()

        return result

    return None
