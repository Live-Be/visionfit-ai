"""Hilfsfunktionen für Session-IDs und -Metadaten."""

import uuid
from datetime import datetime


def new_session_id() -> str:
    """Erzeugt eine neue eindeutige Session-ID."""
    return str(uuid.uuid4())


def current_timestamp() -> str:
    """Gibt den aktuellen Zeitstempel als ISO-String zurück."""
    return datetime.now().isoformat()


def build_session_meta(session_id: str) -> dict:
    """Erstellt ein Basis-Metadaten-Dict für eine Session."""
    return {
        "session_id": session_id,
        "created_at": current_timestamp(),
        "app": "VisionFit AI",
        "version": "0.1.0",
    }
