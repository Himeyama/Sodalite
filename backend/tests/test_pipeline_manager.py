"""Tests for PipelineManager's model loading dispatch logic."""

from unittest.mock import MagicMock, patch

from sodalite_backend.inference.pipeline_manager import PipelineManager, _select_device
from sodalite_backend.schemas.generation import LoraSpec


def _make_manager() -> PipelineManager:
    with patch(
        "sodalite_backend.inference.pipeline_manager.AutoPipelineForText2Image"
    ) as mock_auto:
        mock_auto.from_pretrained.return_value.to.return_value = MagicMock()
        manager = PipelineManager()
        manager.load_initial_model("stub/model")
        return manager


def test_select_device_prefers_cuda_over_directml() -> None:
    with (
        patch("sodalite_backend.inference.pipeline_manager.torch.cuda.is_available", return_value=True),
        patch("sodalite_backend.inference.pipeline_manager.torch.version.hip", None),
        patch("sodalite_backend.inference.pipeline_manager._get_directml_device") as directml,
    ):
        assert _select_device() == ("cuda", "cuda")
        directml.assert_not_called()


def test_select_device_identifies_rocm_build() -> None:
    with (
        patch("sodalite_backend.inference.pipeline_manager.torch.cuda.is_available", return_value=True),
        patch("sodalite_backend.inference.pipeline_manager.torch.version.hip", "7.2.1"),
        patch("sodalite_backend.inference.pipeline_manager._get_directml_device") as directml,
    ):
        assert _select_device() == ("cuda", "rocm")
        directml.assert_not_called()


def test_select_device_uses_directml_when_cuda_is_unavailable() -> None:
    dml_device = MagicMock()
    with (
        patch("sodalite_backend.inference.pipeline_manager.torch.cuda.is_available", return_value=False),
        patch(
            "sodalite_backend.inference.pipeline_manager._get_directml_device",
            return_value=dml_device,
        ),
    ):
        assert _select_device() == (dml_device, "directml")


def test_select_device_falls_back_to_cpu() -> None:
    with (
        patch("sodalite_backend.inference.pipeline_manager.torch.cuda.is_available", return_value=False),
        patch(
            "sodalite_backend.inference.pipeline_manager._get_directml_device", return_value=None
        ),
    ):
        assert _select_device() == ("cpu", "cpu")


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
        patch("sodalite_backend.inference.pipeline_manager.StableDiffusionXLPipeline") as mock_sdxl,
        patch("sodalite_backend.inference.pipeline_manager.StableDiffusionPipeline") as mock_sd15,
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

    list(
        manager.generate(
            prompt="a cat",
            negative_prompt="",
            steps=4,
            cfg_scale=7.0,
            width=64,
            height=64,
            sampler="euler_a",
            seed=None,
            loras=[
                LoraSpec(model_id="a/lora", weight=0.8),
                LoraSpec(model_id="b/lora", weight=0.3),
            ],
        )
    )

    assert pipeline.load_lora_into_unet.call_count == 2
    pipeline.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.8, 0.3])


def test_generate_skips_text_encoder_lora_diffusers_cannot_parse() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    # Mirror the diffusers 0.39 bug where inferring the rank of an
    # unparseable text-encoder sub-weight raises IndexError.
    pipeline.load_lora_into_text_encoder.side_effect = IndexError("list index out of range")
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    list(
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

    list(
        manager.generate(
            prompt="a cat",
            negative_prompt="",
            steps=4,
            cfg_scale=7.0,
            width=64,
            height=64,
            sampler="euler_a",
            seed=None,
            loras=[
                LoraSpec(model_id="bad/lora", weight=0.5),
                LoraSpec(model_id="good/lora", weight=0.9),
            ],
        )
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

    list(
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
    )

    pipeline.set_adapters.assert_not_called()


def test_generate_unloads_loras_after_generation() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    pipeline.lora_state_dict.return_value = ({}, {}, {})
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    list(
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
    )

    pipeline.unload_lora_weights.assert_called_once()


def test_generate_without_loras_does_not_touch_lora_apis() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    list(
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
    )

    pipeline.load_lora_weights.assert_not_called()
    pipeline.set_adapters.assert_not_called()
    pipeline.unload_lora_weights.assert_not_called()


def test_generate_yields_one_image_per_batch_item() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    images = list(
        manager.generate(
            prompt="a cat",
            negative_prompt="",
            steps=4,
            cfg_scale=7.0,
            width=64,
            height=64,
            sampler="euler_a",
            seed=None,
            batch_size=3,
        )
    )

    assert len(images) == 3
    assert pipeline.call_count == 3


def test_generate_uses_img2img_pipeline_for_an_initial_image() -> None:
    manager = _make_manager()
    text_pipeline = MagicMock()
    image_pipeline = MagicMock()
    manager._pipeline = text_pipeline
    manager.set_sampler = MagicMock()

    with patch(
        "sodalite_backend.inference.pipeline_manager.AutoPipelineForImage2Image"
    ) as image_pipeline_factory:
        image_pipeline_factory.from_pipe.return_value = image_pipeline
        list(
            manager.generate(
                prompt="a cat",
                negative_prompt="",
                steps=4,
                cfg_scale=7.0,
                width=64,
                height=64,
                sampler="euler_a",
                seed=None,
                initial_image=MagicMock(),
                strength=0.7,
            )
        )

    image_pipeline_factory.from_pipe.assert_called_once_with(text_pipeline)
    assert image_pipeline.call_args.kwargs["strength"] == 0.7


def test_generate_stops_early_when_should_stop_becomes_true() -> None:
    manager = _make_manager()
    pipeline = MagicMock()
    manager._pipeline = pipeline
    manager.set_sampler = MagicMock()

    stop_after_first = iter([False, True, True, True])

    images = list(
        manager.generate(
            prompt="a cat",
            negative_prompt="",
            steps=4,
            cfg_scale=7.0,
            width=64,
            height=64,
            sampler="euler_a",
            seed=None,
            batch_size=5,
            should_stop=lambda: next(stop_after_first),
        )
    )

    # should_stop is checked before each image; it flips to True before the
    # second image starts, so only the first image is produced.
    assert len(images) == 1
    assert pipeline.call_count == 1
