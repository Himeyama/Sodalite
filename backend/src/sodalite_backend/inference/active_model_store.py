"""Persists the last-selected model id across restarts.

Without this, every restart falls back to the CLI/config default regardless of
what the user last picked in the WinUI3 app or the browser webui. Both share
this store (via [data_dir]) so switching in either surfaces on the next launch
of both.
"""

import json

from sodalite_backend.data_dir import data_path

STORE_NAME = "active_model.json"


def load_active_model_id() -> str | None:
    store_path = data_path(STORE_NAME)
    if not store_path.exists():
        return None

    with store_path.open(encoding="utf-8") as file:
        data = json.load(file)

    return data.get("model_id")


def save_active_model_id(model_id: str) -> None:
    with data_path(STORE_NAME).open("w", encoding="utf-8") as file:
        json.dump({"model_id": model_id}, file, indent=2, ensure_ascii=False)
