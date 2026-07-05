"""Tests for the generation and system API endpoints."""

from fastapi.testclient import TestClient

from sodalite_backend.schemas.generation import ModelInfo


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "device": "cpu", "loaded_model": "stub/model"}


def test_samplers(client: TestClient) -> None:
    response = client.get("/api/v1/samplers")
    assert response.status_code == 200
    assert "euler_a" in response.json()


def test_text_to_image_returns_completed_job(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generations/text-to-image",
        json={"prompt": "a cat wearing sunglasses", "steps": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["image_url"].startswith("/api/v1/images/")


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


def test_import_model(client: TestClient, tmp_path) -> None:
    checkpoint = tmp_path / "my-model.safetensors"
    checkpoint.write_bytes(b"fake checkpoint data")

    response = client.post("/api/v1/models/imported", json={"model_path": str(checkpoint)})

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == str(checkpoint)
    assert body["is_active"] is False
    assert body["size_on_disk_bytes"] == checkpoint.stat().st_size


def test_import_model_rejects_missing_file(client: TestClient, tmp_path) -> None:
    missing = tmp_path / "does-not-exist.safetensors"

    response = client.post("/api/v1/models/imported", json={"model_path": str(missing)})

    assert response.status_code == 422


def test_import_model_rejects_unsupported_extension(client: TestClient, tmp_path) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a checkpoint")

    response = client.post("/api/v1/models/imported", json={"model_path": str(unsupported)})

    assert response.status_code == 422


def test_list_models_includes_imported_models(client: TestClient, tmp_path) -> None:
    checkpoint = tmp_path / "my-model.safetensors"
    checkpoint.write_bytes(b"fake checkpoint data")
    client.post("/api/v1/models/imported", json={"model_path": str(checkpoint)})

    response = client.get("/api/v1/models")

    assert response.status_code == 200
    model_ids = [model["model_id"] for model in response.json()]
    assert str(checkpoint) in model_ids


def test_remove_imported_model(client: TestClient, tmp_path) -> None:
    checkpoint = tmp_path / "my-model.safetensors"
    checkpoint.write_bytes(b"fake checkpoint data")
    client.post("/api/v1/models/imported", json={"model_path": str(checkpoint)})

    response = client.request(
        "DELETE", "/api/v1/models/imported", json={"model_path": str(checkpoint)}
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == str(checkpoint)
    # Dropped from the registry but the file itself is left on disk.
    assert checkpoint.exists()
    model_ids = [model["model_id"] for model in client.get("/api/v1/models").json()]
    assert str(checkpoint) not in model_ids


def test_remove_imported_model_rejects_active_model(
    client: TestClient, tmp_path, mock_pipeline_manager
) -> None:
    checkpoint = tmp_path / "active.safetensors"
    checkpoint.write_bytes(b"fake checkpoint data")
    client.post("/api/v1/models/imported", json={"model_path": str(checkpoint)})
    mock_pipeline_manager.model_id = str(checkpoint)

    response = client.request(
        "DELETE", "/api/v1/models/imported", json={"model_path": str(checkpoint)}
    )

    assert response.status_code == 409


def test_remove_imported_model_rejects_unknown_model(client: TestClient, tmp_path) -> None:
    response = client.request(
        "DELETE",
        "/api/v1/models/imported",
        json={"model_path": str(tmp_path / "never-imported.safetensors")},
    )

    assert response.status_code == 404


def test_import_lora(client: TestClient, tmp_path) -> None:
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"fake lora data")

    response = client.post("/api/v1/loras/imported", json={"lora_path": str(lora)})

    assert response.status_code == 200
    body = response.json()
    assert body["lora_id"] == str(lora)
    assert body["size_on_disk_bytes"] == lora.stat().st_size


def test_import_lora_rejects_missing_file(client: TestClient, tmp_path) -> None:
    missing = tmp_path / "does-not-exist.safetensors"

    response = client.post("/api/v1/loras/imported", json={"lora_path": str(missing)})

    assert response.status_code == 422


def test_import_lora_rejects_unsupported_extension(client: TestClient, tmp_path) -> None:
    unsupported = tmp_path / "style.ckpt"
    unsupported.write_bytes(b"not a lora")

    response = client.post("/api/v1/loras/imported", json={"lora_path": str(unsupported)})

    assert response.status_code == 422


def test_list_loras_includes_imported_loras(client: TestClient, tmp_path) -> None:
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"fake lora data")
    client.post("/api/v1/loras/imported", json={"lora_path": str(lora)})

    response = client.get("/api/v1/loras")

    assert response.status_code == 200
    lora_ids = [item["lora_id"] for item in response.json()]
    assert str(lora) in lora_ids


def test_text_to_image_forwards_loras(client: TestClient, mock_pipeline_manager) -> None:
    response = client.post(
        "/api/v1/generations/text-to-image",
        json={
            "prompt": "a cat",
            "steps": 4,
            "loras": [{"model_id": "some/lora", "weight": 0.8}],
        },
    )

    assert response.status_code == 200
    _, kwargs = mock_pipeline_manager.generate.call_args
    assert [lora.model_id for lora in kwargs["loras"]] == ["some/lora"]
    assert [lora.weight for lora in kwargs["loras"]] == [0.8]
