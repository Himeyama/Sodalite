"""Output directory management for generated images.

The output directory lives under the shared app data directory (see
[data_dir]) so the WinUI app and the browser webui show the same gallery
regardless of each process's current working directory.
"""

import uuid
from pathlib import Path

from sodalite_backend.data_dir import data_dir


def output_dir() -> Path:
    """Return the image output directory, creating it if needed."""
    path = data_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_image_path() -> Path:
    return output_dir() / f"{uuid.uuid4().hex}.png"
