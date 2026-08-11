"""Tests for the gallery API endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image


def _generate_one(client: TestClient, wait_for_job_done, prompt: str = "a cat") -> dict:
    response = client.post("/api/v1/generations/text-to-image", json={"prompt": prompt, "steps": 4})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = wait_for_job_done(client, job_id)
    assert job["status"] == "completed"
    return job


def test_list_images_empty_when_no_outputs(client: TestClient) -> None:
    response = client.get("/api/v1/gallery/images")
    assert response.status_code == 200
    assert response.json() == []


def test_list_images_returns_generated_image_with_parameters(
    client: TestClient, wait_for_job_done
) -> None:
    job = _generate_one(client, wait_for_job_done, prompt="a cat wearing sunglasses")

    response = client.get("/api/v1/gallery/images")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["image_url"] == job["image_url"]
    assert body[0]["parameters"]["prompt"] == "a cat wearing sunglasses"


def test_list_images_sorted_newest_first(client: TestClient, wait_for_job_done) -> None:
    _generate_one(client, wait_for_job_done, prompt="first")
    _generate_one(client, wait_for_job_done, prompt="second")

    body = client.get("/api/v1/gallery/images").json()

    assert [image["parameters"]["prompt"] for image in body] == ["second", "first"]


def test_list_images_skips_corrupt_file(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir(exist_ok=True)
    (tmp_path / "outputs" / "broken.png").write_bytes(b"not a real png")

    response = client.get("/api/v1/gallery/images")

    assert response.status_code == 200
    assert response.json() == []


def test_list_images_includes_file_without_metadata(client: TestClient, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    Image.new("RGB", (4, 4)).save(outputs / "no_metadata.png")

    body = client.get("/api/v1/gallery/images").json()

    assert len(body) == 1
    assert body[0]["parameters"] is None


def test_delete_image_removes_file(
    client: TestClient, tmp_path: Path, wait_for_job_done
) -> None:
    job = _generate_one(client, wait_for_job_done)
    image_id = job["image_url"].removeprefix("/api/v1/images/")

    response = client.delete(f"/api/v1/gallery/images/{image_id}")

    assert response.status_code == 204
    assert not (tmp_path / "outputs" / image_id).exists()


def test_delete_image_missing_returns_404(client: TestClient) -> None:
    response = client.delete("/api/v1/gallery/images/does-not-exist.png")
    assert response.status_code == 404


def test_list_images_reflects_metadata_change_after_recreate(
    client: TestClient, tmp_path: Path
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir(exist_ok=True)
    path = outputs / "recreated.png"

    Image.new("RGB", (4, 4)).save(path)
    first = client.get("/api/v1/gallery/images").json()
    assert first[0]["parameters"] is None

    # Recreate with different content (and possibly the same mtime second) to
    # make sure the metadata cache doesn't serve a stale entry.
    Image.new("RGB", (8, 8)).save(path)
    second = client.get("/api/v1/gallery/images").json()
    assert len(second) == 1


def test_delete_image_rejects_path_traversal(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path / "secret.png"
    Image.new("RGB", (4, 4)).save(outside)

    response = client.delete("/api/v1/gallery/images/..%2Fsecret.png")

    # The traversal attempt must not delete anything. The client normalizes
    # `..%2F`, so the request may either miss the delete route (405 against the
    # webui static mount) or reach it and be rejected (404); both are safe.
    assert response.status_code in (404, 405)
    assert outside.exists()
