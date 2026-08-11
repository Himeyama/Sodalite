"""Shared fixtures for the API test suite."""

import time
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_pipeline_manager() -> MagicMock:
    manager = MagicMock()
    manager.device = "cpu"
    manager.model_id = "stub/model"
    manager.is_ready = True
    manager.generate.return_value = [Image.new("RGB", (8, 8))]
    return manager


@pytest.fixture
def client(mock_pipeline_manager: MagicMock, tmp_path, monkeypatch) -> Iterator[TestClient]:
    # Point the shared data dir (gallery outputs, settings, known models) at a
    # temp dir so each test is isolated. output_dir() resolves to tmp_path/outputs.
    monkeypatch.setenv("SODALITE_DATA_DIR", str(tmp_path))

    from sodalite_backend import main as main_module

    # main.py constructs PipelineManager() with no args, then calls
    # load_initial_model(model_id) on a background thread. Tests need the
    # model to appear ready synchronously, so the mock's load_initial_model
    # is a no-op and is_ready/model_id are already set above.
    monkeypatch.setattr(main_module, "PipelineManager", lambda: mock_pipeline_manager)

    app = main_module.create_app("stub/model")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def wait_for_job_done():
    """Poll a generation job until it leaves queued/running, for use in tests.

    Generation now runs on a background thread, so a freshly-started job may
    still be `queued` the instant the POST response comes back.
    """

    def _wait(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        job = client.get(f"/api/v1/generations/{job_id}").json()
        while job["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.01)
            job = client.get(f"/api/v1/generations/{job_id}").json()
        return job

    return _wait
