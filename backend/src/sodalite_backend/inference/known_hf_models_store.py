"""Persists the set of Hugging Face repo ids the app should surface as models.

The HF cache can hold many repos the user never picked as a base model (sub-models
pulled in as dependencies, leftovers from unrelated tools). Rather than scanning the
whole cache, the app tracks the repos it has actually offered — the built-in default
plus any the user has activated — so the model list stays focused.
"""

import json

from sodalite_backend.data_dir import data_path

STORE_NAME = "known_hf_models.json"


def load_known_hf_model_ids() -> list[str]:
    store_path = data_path(STORE_NAME)
    if not store_path.exists():
        return []

    with store_path.open(encoding="utf-8") as file:
        return json.load(file)


def add_known_hf_model_id(model_id: str) -> list[str]:
    ids = load_known_hf_model_ids()
    if model_id not in ids:
        ids.append(model_id)
        _save(ids)
    return ids


def remove_known_hf_model_id(model_id: str) -> list[str]:
    ids = load_known_hf_model_ids()
    if model_id in ids:
        ids.remove(model_id)
        _save(ids)
    return ids


def _save(ids: list[str]) -> None:
    with data_path(STORE_NAME).open("w", encoding="utf-8") as file:
        json.dump(ids, file, indent=2, ensure_ascii=False)
