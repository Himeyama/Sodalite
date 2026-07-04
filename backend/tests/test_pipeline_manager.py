"""Tests for PipelineManager's model loading dispatch logic."""

from unittest.mock import MagicMock, patch

from sdapp_backend.inference.pipeline_manager import PipelineManager


def _make_manager() -> PipelineManager:
    with patch(
        "sdapp_backend.inference.pipeline_manager.AutoPipelineForText2Image"
    ) as mock_auto:
        mock_auto.from_pretrained.return_value.to.return_value = MagicMock()
        return PipelineManager("stub/model")


def test_load_model_uses_from_pretrained_for_repo_id() -> None:
    manager = _make_manager()

    with patch(
        "sdapp_backend.inference.pipeline_manager.AutoPipelineForText2Image"
    ) as mock_auto:
        mock_auto.from_pretrained.return_value.to.return_value = MagicMock()

        manager.load_model("other/model")

        mock_auto.from_pretrained.assert_called_once()
        assert manager.model_id == "other/model"


def test_load_model_uses_single_file_for_local_checkpoint(tmp_path) -> None:
    manager = _make_manager()
    checkpoint = tmp_path / "my-model.safetensors"
    checkpoint.write_bytes(b"fake checkpoint data")

    with patch(
        "sdapp_backend.inference.pipeline_manager.StableDiffusionXLPipeline"
    ) as mock_sdxl:
        mock_sdxl.from_single_file.return_value.to.return_value = MagicMock()

        manager.load_model(str(checkpoint))

        mock_sdxl.from_single_file.assert_called_once()
        assert manager.model_id == str(checkpoint)


def test_load_model_falls_back_to_sd15_when_sdxl_load_fails(tmp_path) -> None:
    manager = _make_manager()
    checkpoint = tmp_path / "my-model.safetensors"
    checkpoint.write_bytes(b"fake checkpoint data")

    with (
        patch(
            "sdapp_backend.inference.pipeline_manager.StableDiffusionXLPipeline"
        ) as mock_sdxl,
        patch(
            "sdapp_backend.inference.pipeline_manager.StableDiffusionPipeline"
        ) as mock_sd15,
    ):
        mock_sdxl.from_single_file.side_effect = RuntimeError("not an sdxl checkpoint")
        mock_sd15.from_single_file.return_value.to.return_value = MagicMock()

        manager.load_model(str(checkpoint))

        mock_sd15.from_single_file.assert_called_once()
        assert manager.model_id == str(checkpoint)
