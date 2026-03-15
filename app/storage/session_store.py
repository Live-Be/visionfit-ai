"""Session-Speicherung als JSON-Dateien."""

import json
from pathlib import Path
from datetime import datetime

from app.utils.config import SESSION_DIR


def _ensure_session_dir(session_dir: str = SESSION_DIR) -> Path:
    """Stellt sicher, dass der Session-Ordner existiert."""
    path = Path(session_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_session(session_data: dict, session_dir: str = SESSION_DIR) -> str:
    """Speichert eine Session als JSON-Datei.

    Args:
        session_data: Dictionary mit allen Session-Daten.
        session_dir:  Zielverzeichnis für JSON-Dateien.

    Returns:
        Absoluter Dateipfad der gespeicherten JSON-Datei.
    """
    dir_path = _ensure_session_dir(session_dir)

    session_id = session_data.get("session_id", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{session_id[:8]}_{timestamp}.json"
    file_path = dir_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return str(file_path.resolve())


def load_session(file_path: str) -> dict:
    """Lädt eine gespeicherte Session aus einer JSON-Datei.

    Args:
        file_path: Pfad zur JSON-Datei.

    Returns:
        Dictionary mit den Session-Daten.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
