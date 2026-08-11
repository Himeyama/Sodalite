"""Tests for the generation and system API endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from sodalite_backend.schemas.generation import ModelInfo


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "device": "cpu",
        "loaded_model": "stub/model",
        "model_ready": True,
    }


def test_samplers(client: TestClient) -> None:
    response = client.get("/api/v1/samplers")
    assert response.status_code == 200
    assert "euler_a" in response.json()


def test_text_to_image_starts_a_queued_job(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generations/text-to-image",
        json={"prompt": "a cat wearing sunglasses", "steps": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"]
    assert body["status"] in ("queued", "running", "completed")


def test_text_to_image_job_completes_with_image(client: TestClient, wait_for_job_done) -> None:
    response = client.post(
        "/api/v1/generations/text-to-image",
        json={"prompt": "a cat wearing sunglasses", "steps": 4},
    )
    job_id = response.json()["job_id"]

    body = wait_for_job_done(client, job_id)

    assert body["status"] == "completed"
    assert body["image_url"].startswith("/api/v1/images/")
    assert Path(body["image_path"]).is_absolute()
    assert Path(body["image_path"]).name == body["image_url"].removeprefix("/api/v1/images/")


def test_get_generation_job_missing_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/generations/no-such-job")
    assert response.status_code == 404


def test_cancel_generation_job_missing_returns_404(client: TestClient) -> None:
    response = client.delete("/api/v1/generations/no-such-job")
    assert response.status_code == 404


def test_text_to_image_rejects_invalid_dimensions(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generations/text-to-image",
        json={"prompt": "test", "width": 10},
    )
    assert response.status_code == 422


def test_list_models(client: TestClient, monkeypatch) -> None:
    fake_models = [
        ModelInfo(model_id="stub/model", is_active=True, size_on_disk_bytes=123),
        ModelInfo(model_id="other/model", is_active=False, size_on_disk_bytes=456),
    ]
    monkeypatch.setattr(
        "sodalite_backend.api.system.list_cached_models", lambda active_model_id: fake_models
    )

    response = client.get("/api/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body == [model.model_dump() for model in fake_models]


def test_set_active_model(client: TestClient, mock_pipeline_manager) -> None:
    def load_model(model_id: str) -> None:
        mock_pipeline_manager.model_id = model_id

    mock_pipeline_manager.load_model.side_effect = load_model

    response = client.post("/api/v1/models/active", json={"model_id": "other/model"})

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "other/model",
        "is_active": True,
        "size_on_disk_bytes": 0,
    }
    mock_pipeline_manager.load_model.assert_called_once_with("other/model")


def test_set_active_model_rejects_unknown_model(
    client: TestClient, mock_pipeline_manager
) -> None:
    mock_pipeline_manager.load_model.side_effect = OSError("model not found")

    response = client.post("/api/v1/models/active", json={"model_id": "no/such-model"})

    assert response.status_code == 422


def test_get_directories_defaults_to_unset(client: TestClient) -> None:
    response = client.get("/api/v1/settings/directories")

    assert response.status_code == 200
    assert response.json() == {"model_dir": None, "lora_dir": None}


def test_put_directories_persists_and_round_trips(client: TestClient, tmp_path) -> None:
    model_dir = tmp_path / "models"
    lora_dir = tmp_path / "loras"
    model_dir.mkdir()
    lora_dir.mkdir()

    put = client.put(
        "/api/v1/settings/directories",
        json={"model_dir": str(model_dir), "lora_dir": str(lora_dir)},
    )
    assert put.status_code == 200
    assert put.json() == {"model_dir": str(model_dir), "lora_dir": str(lora_dir)}

    got = client.get("/api/v1/settings/directories")
    assert got.json() == {"model_dir": str(model_dir), "lora_dir": str(lora_dir)}


def test_put_directories_rejects_missing_directory(client: TestClient, tmp_path) -> None:
    response = client.put(
        "/api/v1/settings/directories",
        json={"model_dir": str(tmp_path / "nope"), "lora_dir": None},
    )

    assert response.status_code == 422


def test_list_models_scans_model_directory(client: TestClient, tmp_path) -> None:
    model_dir = tmp_path / "models"
    (model_dir / "sub").mkdir(parents=True)
    (model_dir / "a.safetensors").write_bytes(b"x")
    (model_dir / "sub" / "b.ckpt").write_bytes(b"y")
    (model_dir / "notes.txt").write_text("ignore me")
    client.put(
        "/api/v1/settings/directories", json={"model_dir": str(model_dir), "lora_dir": None}
    )

    model_ids = [model["model_id"] for model in client.get("/api/v1/models").json()]

    assert str(model_dir / "a.safetensors") in model_ids
    assert str(model_dir / "sub" / "b.ckpt") not in model_ids
    assert str(model_dir / "notes.txt") not in model_ids


def test_list_loras_scans_lora_directory(client: TestClient, tmp_path) -> None:
    lora_dir = tmp_path / "loras"
    (lora_dir / "deep").mkdir(parents=True)
    (lora_dir / "l1.safetensors").write_bytes(b"z")
    (lora_dir / "deep" / "l2.safetensors").write_bytes(b"w")
    (lora_dir / "l3.ckpt").write_bytes(b"q")  # .ckpt is not a LoRA format
    client.put(
        "/api/v1/settings/directories", json={"model_dir": None, "lora_dir": str(lora_dir)}
    )

    lora_ids = [item["lora_id"] for item in client.get("/api/v1/loras").json()]

    assert lora_ids == [str(lora_dir / "l1.safetensors")]


def test_list_loras_empty_when_unset(client: TestClient) -> None:
    response = client.get("/api/v1/loras")

    assert response.status_code == 200
    assert response.json() == []


def test_text_to_image_forwards_loras(
    client: TestClient, mock_pipeline_manager, wait_for_job_done
) -> None:
    response = client.post(
        "/api/v1/generations/text-to-image",
        json={
            "prompt": "a cat",
            "steps": 4,
            "loras": [{"model_id": "some/lora", "weight": 0.8}],
        },
    )
    job_id = response.json()["job_id"]

    assert wait_for_job_done(client, job_id)["status"] == "completed"
    _, kwargs = mock_pipeline_manager.generate.call_args
    assert [lora.model_id for lora in kwargs["loras"]] == ["some/lora"]
    assert [lora.weight for lora in kwargs["loras"]] == [0.8]
