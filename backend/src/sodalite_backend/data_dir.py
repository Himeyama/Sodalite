"""Resolves the application data directory (gallery outputs, settings, known models).

All persistent state lives in a single fixed location, ``%LOCALAPPDATA%\\Sodalite``,
regardless of how or from where the backend is launched. Anchoring to one absolute
path — rather than the process CWD or the backend project directory — is what makes
the WinUI3 app and the browser webui share the *same* gallery, model list and folder
settings across every configuration (dev run, installed build, either frontend).

``SODALITE_DATA_DIR`` overrides the base (used by tests to isolate state, and
available as an escape hatch).
"""

import os
from pathlib import Path


def data_dir() -> Path:
    """Return the app data directory, creating it if needed."""
    override = os.environ.get("SODALITE_DATA_DIR")
    if override:
        base = Path(override)
    else:
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) / "Sodalite" if local_app_data else Path.home() / ".sodalite"

    base.mkdir(parents=True, exist_ok=True)
    return base


def data_path(name: str) -> Path:
    """Return the path to a file named ``name`` inside the data directory."""
    return data_dir() / name
