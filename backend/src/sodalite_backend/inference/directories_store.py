"""Persists the user-chosen model and LoRA directories across restarts.

The app scans these directories for checkpoints and LoRA files rather than
tracking individually imported files. Both are optional; when unset the
corresponding scan simply yields nothing.
"""

import json
from dataclasses import dataclass

from sodalite_backend.data_dir import data_path

STORE_NAME = "directories.json"


@dataclass(frozen=True, slots=True)
class ScanDirectories:
    model_dir: str | None
    lora_dir: str | None


def load_directories() -> ScanDirectories:
    store_path = data_path(STORE_NAME)
    if not store_path.exists():
        return ScanDirectories(model_dir=None, lora_dir=None)

    with store_path.open(encoding="utf-8") as file:
        data = json.load(file)

    return ScanDirectories(
        model_dir=data.get("model_dir"),
        lora_dir=data.get("lora_dir"),
    )


def save_directories(directories: ScanDirectories) -> ScanDirectories:
    with data_path(STORE_NAME).open("w", encoding="utf-8") as file:
        json.dump(
            {"model_dir": directories.model_dir, "lora_dir": directories.lora_dir},
            file,
            indent=2,
            ensure_ascii=False,
        )
    return directories
