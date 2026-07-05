"""Tests for model listing: only known HF repos and imported files are surfaced."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from sodalite_backend.inference import known_hf_models_store, model_registry


@dataclass
class _FakeFile:
    file_name: str


@dataclass
class _FakeRevision:
    files: list[_FakeFile]


@dataclass
class _FakeRepo:
    repo_id: str
    repo_type: str
    size_on_disk: int
    revisions: list[_FakeRevision]


def _pipeline_repo(repo_id: str, size: int = 100) -> _FakeRepo:
    return _FakeRepo(
        repo_id=repo_id,
        repo_type="model",
        size_on_disk=size,
        revisions=[_FakeRevision(files=[_FakeFile("model_index.json")])],
    )


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _patch_cache(*repos: _FakeRepo):
    cache = MagicMock()
    cache.repos = list(repos)
    return patch(
        "sodalite_backend.inference.model_registry.scan_cache_dir", return_value=cache
    )


def test_only_known_hf_repos_are_listed() -> None:
    known_hf_models_store.add_known_hf_model_id("stabilityai/sd-turbo")

    cached = (
        _pipeline_repo("stabilityai/sd-turbo"),
        _pipeline_repo("stable-diffusion-v1-5/stable-diffusion-v1-5"),
        _pipeline_repo("stabilityai/stable-diffusion-xl-base-1.0"),
    )
    with _patch_cache(*cached):
        models = model_registry.list_cached_models("stabilityai/sd-turbo")

    model_ids = [model.model_id for model in models]
    assert model_ids == ["stabilityai/sd-turbo"]


def test_active_model_is_listed_even_if_not_yet_known() -> None:
    # Nothing recorded yet; the active model must still appear so the UI can flag it.
    with _patch_cache(_pipeline_repo("some/model")):
        models = model_registry.list_cached_models("some/model")

    active = [model for model in models if model.is_active]
    assert [model.model_id for model in active] == ["some/model"]
