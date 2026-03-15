"""Konfigurationsmodul – lädt Umgebungsvariablen via python-dotenv."""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env aus dem Projektstamm laden (falls vorhanden)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


def get(key: str, default: str = "") -> str:
    """Gibt den Wert einer Umgebungsvariable zurück."""
    return os.getenv(key, default)


APP_ENV: str = get("APP_ENV", "development")
APP_NAME: str = get("APP_NAME", "VisionFit AI")
SESSION_DIR: str = get("SESSION_DIR", "data/sessions")
