"""Tests for PipelineManager's model loading dispatch logic."""

from unittest.mock import MagicMock, patch

from sodalite_backend.inference.pipeline_manager import PipelineManager
from sodalite_backend.schemas.generation import LoraSpec


def _make_manager() -> PipelineManager:
    with patch(
        "sodalite_backend.inference.pipeline_manager.AutoPipelineForText2Image"
    ) as mock_auto:
        mock_auto.from_pretrained.return_value.to.return_value = MagicMock()
        return PipelineManager("stub/model")


def test_load_model_uses_from_pretrained_for_repo_id() -> None:
    manager = _make_manager()

    with patch(
        "sodalite_backend.inference.pipeline_manager.AutoPipelineForText2Image"
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
        "sodalite_backend.inference.pipeline_manager.StableDiffusionXLPipeline"
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
            "sodalite_backend.inference.pipeline_manager.StableDiffusionXLPipeline"
        ) as mock_sdxl,
        patch(
            "sodalite_backend.inference.pipeline_manager.StableDiffusionPipeline"
        ) as mock_sd15,
    ):
        mock_sdxl.from_single_file.side_effect = RuntimeError("not an sdxl checkpoint")
        mock_sd15.from_single_file.return_value.to.return_value = MagicMock()

        manager.load_model(str(checkpoint))

        mock_sd15.from_single_file.assert_called_once()
        assert manager.model_id == str(checkpoint)


def test_generate_loads_and_activates_loras_by_weight() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    manager.generate(
        prompt="a cat",
        negative_prompt="",
        steps=4,
        cfg_scale=7.0,
        width=64,
        height=64,
        sampler="euler_a",
        seed=None,
        loras=[LoraSpec(model_id="a/lora", weight=0.8), LoraSpec(model_id="b/lora", weight=0.3)],
    )

    assert pipeline.load_lora_into_unet.call_count == 2
    pipeline.set_adapters.assert_called_once_with(
        ["lora_0", "lora_1"], adapter_weights=[0.8, 0.3]
    )


def test_generate_skips_text_encoder_lora_diffusers_cannot_parse() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    # Mirror the diffusers 0.39 bug where inferring the rank of an
    # unparseable text-encoder sub-weight raises IndexError.
    pipeline.load_lora_into_text_encoder.side_effect = IndexError("list index out of range")
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    manager.generate(
        prompt="a cat",
        negative_prompt="",
        steps=4,
        cfg_scale=7.0,
        width=64,
        height=64,
        sampler="euler_a",
        seed=None,
        loras=[LoraSpec(model_id="a/lora", weight=1.0)],
    )

    # The UNet still loads and the adapter is activated despite the encoder failure.
    pipeline.load_lora_into_unet.assert_called_once()
    pipeline.set_adapters.assert_called_once_with(["lora_0"], adapter_weights=[1.0])


def test_generate_skips_incompatible_lora_but_applies_the_rest() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    # First LoRA is an architecture mismatch (e.g. SDXL LoRA on an SD1.5 model);
    # diffusers rejects it while loading into the UNet.
    pipeline.load_lora_into_unet.side_effect = [ValueError("size mismatch"), None]
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    manager.generate(
        prompt="a cat",
        negative_prompt="",
        steps=4,
        cfg_scale=7.0,
        width=64,
        height=64,
        sampler="euler_a",
        seed=None,
        loras=[LoraSpec(model_id="bad/lora", weight=0.5), LoraSpec(model_id="good/lora", weight=0.9)],
    )

    # The bad adapter is dropped and only the compatible one is activated.
    pipeline.delete_adapters.assert_called_once_with("lora_0")
    pipeline.set_adapters.assert_called_once_with(["lora_1"], adapter_weights=[0.9])


def test_generate_does_not_activate_adapters_when_all_loras_incompatible() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    pipeline.load_lora_into_unet.side_effect = ValueError("size mismatch")
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    manager.generate(
        prompt="a cat",
        negative_prompt="",
        steps=4,
        cfg_scale=7.0,
        width=64,
        height=64,
        sampler="euler_a",
        seed=None,
        loras=[LoraSpec(model_id="bad/lora", weight=1.0)],
    )

    pipeline.set_adapters.assert_not_called()


def test_generate_unloads_loras_after_generation() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    manager.generate(
        prompt="a cat",
        negative_prompt="",
        steps=4,
        cfg_scale=7.0,
        width=64,
        height=64,
        sampler="euler_a",
        seed=None,
        loras=[LoraSpec(model_id="a/lora", weight=1.0)],
    )

    pipeline.unload_lora_weights.assert_called_once()


def test_generate_without_loras_does_not_touch_lora_apis() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    manager.generate(
        prompt="a cat",
        negative_prompt="",
        steps=4,
        cfg_scale=7.0,
        width=64,
        height=64,
        sampler="euler_a",
        seed=None,
    )

    pipeline.load_lora_weights.assert_not_called()
    pipeline.set_adapters.assert_not_called()
    pipeline.unload_lora_weights.assert_not_called()
