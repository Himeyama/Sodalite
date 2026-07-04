"""Persists paths to locally imported single-file checkpoints across restarts."""

import json
from pathlib import Path

STORE_PATH = Path("imported_models.json")


def load_imported_model_paths() -> list[str]:
    if not STORE_PATH.exists():
        return []

    with STORE_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def add_imported_model_path(model_path: str) -> list[str]:
    paths = load_imported_model_paths()
    if model_path not in paths:
        paths.append(model_path)
        _save(paths)
    return paths


def _save(paths: list[str]) -> None:
    with STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(paths, file, indent=2, ensure_ascii=False)
