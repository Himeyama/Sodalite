"""Regression tests for GPU residency, prompt reuse and exact-latent VAE retry."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers import Krea2Pipeline, Krea2Transformer2DModel

from sodalite_backend.inference.krea2_runtime import GpuKrea2Pipeline, _GpuRuntime


def _pipeline(device: str = "cpu") -> GpuKrea2Pipeline:
    pipeline = object.__new__(GpuKrea2Pipeline)
    pipeline.register_to_config(is_distilled=True)
    pipeline.transformer = MagicMock()
    pipeline.text_encoder = MagicMock()
    pipeline.vae = MagicMock()
    pipeline._gpu_runtime = _GpuRuntime(torch.device(device), pipeline.vae.decode)
    return pipeline


def test_reuses_prompt_but_reencodes_changes_in_text_or_length() -> None:
    pipeline = _pipeline()
    embeds, mask = torch.randn(1, 8, 12, 4), torch.ones(1, 8, dtype=torch.bool)
    with (
        patch.object(
            Krea2Pipeline, "get_text_hidden_states", return_value=(embeds, mask)
        ) as encode,
        patch("torch.cuda.empty_cache"),
    ):
        first = pipeline.get_text_hidden_states("a fox", 8)
        second = pipeline.get_text_hidden_states(["a fox"], 8)
        assert encode.call_count == 1
        assert first[0].data_ptr() == second[0].data_ptr()
        pipeline.get_text_hidden_states("a fox", 16)
        pipeline.get_text_hidden_states("a cat", 16)
        assert encode.call_count == 3
    # CPU is used for weight storage after every actual text encoding.
    assert pipeline.text_encoder.to.call_args.args == ("cpu",)


def test_failed_encoding_releases_encoder_and_does_not_cache_partial_result() -> None:
    pipeline = _pipeline()
    with (
        patch.object(
            Krea2Pipeline, "get_text_hidden_states", side_effect=RuntimeError("encoding failed")
        ),
        patch("torch.cuda.empty_cache"),
        pytest.raises(RuntimeError, match="encoding failed"),
    ):
        pipeline.get_text_hidden_states("a fox")
    assert pipeline._gpu_runtime.cache is None
    assert pipeline._gpu_runtime.cache_key is None
    assert pipeline.text_encoder.to.call_args.args == ("cpu",)


@pytest.mark.parametrize("free_gib,group_offload", [(48, False), (32, True), (24, True)])
def test_selects_strategy_from_available_memory(free_gib: int, group_offload: bool) -> None:
    pipeline = _pipeline()
    pipeline._gpu_runtime = None
    # Meta tensors test a realistic 24 GiB model without allocating real memory.
    pipeline.transformer.parameters.return_value = iter(
        [torch.empty(12 * 1024**3, device="meta", dtype=torch.bfloat16)]
    )
    with patch("torch.cuda.mem_get_info", return_value=(free_gib * 1024**3, free_gib * 1024**3)):
        pipeline.configure_gpu("cuda:0")
    assert pipeline._gpu_runtime.group_offload is group_offload
    if group_offload:
        pipeline.transformer.enable_group_offload.assert_called_once()
        kwargs = pipeline.transformer.enable_group_offload.call_args.kwargs
        assert kwargs["onload_device"] == torch.device("cuda:0")
        assert kwargs["num_blocks_per_group"] == 2
    else:
        pipeline.transformer.enable_group_offload.assert_not_called()


def test_vae_oom_retries_same_latents_on_gpu_and_remembers_memory_limit() -> None:
    pipeline = _pipeline("cuda:0")
    latents = torch.randn(1, 16, 1, 8, 8)
    expected = (torch.randn(1, 3, 1, 64, 64),)
    pipeline._gpu_runtime.decode.side_effect = [
        torch.cuda.OutOfMemoryError("decoder"),
        expected,
        expected,
    ]
    with patch("torch.cuda.empty_cache"):
        assert pipeline._decode_on_gpu(latents, return_dict=False) is expected
        assert pipeline._decode_on_gpu(latents, return_dict=False) is expected
    assert pipeline._gpu_runtime.offload_for_decode is True
    assert all(call.args[0] is latents for call in pipeline._gpu_runtime.decode.call_args_list)
    assert all(call.args[0] == torch.device("cuda:0") for call in pipeline.vae.to.call_args_list)
    assert pipeline.transformer.to.call_count == 2


def test_successful_decode_keeps_transformer_resident() -> None:
    pipeline = _pipeline()
    pipeline._decode_on_gpu(torch.zeros(1), return_dict=False)
    pipeline.transformer.to.assert_not_called()
    with patch.object(Krea2Pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()
    pipeline.transformer.to.assert_not_called()
    pipeline.vae.to.assert_called_with("cpu")


def test_releasing_pipeline_invalidates_cache() -> None:
    pipeline = _pipeline()
    pipeline._gpu_runtime.cache = (torch.ones(1), torch.ones(1))
    pipeline._gpu_runtime.cache_key = (("fox",), 512)
    with patch("torch.cuda.empty_cache"):
        pipeline.release_gpu()
    assert pipeline._gpu_runtime.cache is None
    assert pipeline._gpu_runtime.cache_key is None
    pipeline.transformer.to.assert_called_once_with("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA/ROCm GPU")
def test_resident_components_compute_on_real_gpu() -> None:
    pipeline = _pipeline("cuda:0")
    pipeline.transformer = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    pipeline.text_encoder = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    pipeline.vae = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
    pipeline._gpu_runtime.decode = lambda z, **_: (pipeline.vae(z),)
    inputs = torch.ones(1, 4, device="cuda:0", dtype=torch.bfloat16)

    def encode(*_):
        assert next(pipeline.text_encoder.parameters()).device.type == "cuda"
        return pipeline.text_encoder(inputs), torch.ones(1, 4, device="cuda:0", dtype=torch.bool)

    with patch.object(Krea2Pipeline, "get_text_hidden_states", side_effect=encode):
        embeds, _ = pipeline.get_text_hidden_states("fox", 4)
    assert embeds.device.type == "cuda"
    assert next(pipeline.text_encoder.parameters()).device.type == "cpu"
    with patch.object(Krea2Pipeline, "encode_prompt", return_value=(embeds, torch.ones(1))):
        pipeline.encode_prompt(prompt="fox")
    assert pipeline.transformer(inputs).device.type == "cuda"
    assert pipeline._decode_on_gpu(inputs)[0].device.type == "cuda"
    with patch.object(Krea2Pipeline, "maybe_free_model_hooks"):
        pipeline.maybe_free_model_hooks()
    assert next(pipeline.transformer.parameters()).device.type == "cuda"
    with patch("torch.cuda.empty_cache"):
        pipeline.release_gpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA/ROCm GPU")
def test_group_offload_matches_resident_transformer_on_real_gpu() -> None:
    config = dict(
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        text_hidden_dim=32,
        num_text_layers=2,
        text_num_attention_heads=4,
        text_num_key_value_heads=4,
        text_intermediate_size=64,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(8, 4, 4),
    )
    resident = Krea2Transformer2DModel(**config).to("cuda", dtype=torch.bfloat16).eval()
    streamed = Krea2Transformer2DModel(**config).to(dtype=torch.bfloat16).eval()
    streamed.load_state_dict({key: value.cpu() for key, value in resident.state_dict().items()})
    arguments = dict(
        hidden_states=torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, 4, 2, 32, device="cuda", dtype=torch.bfloat16),
        timestep=torch.ones(1, device="cuda", dtype=torch.bfloat16),
        position_ids=torch.zeros(12, 3, device="cuda", dtype=torch.long),
        # Preserve a real padding mask so this test isolates offloading from
        # the independent optimization that selects a different fused kernel.
        encoder_attention_mask=torch.tensor([[True, False, True, True]], device="cuda"),
    )
    with torch.no_grad():
        expected = resident(**arguments).sample
    pipeline = _pipeline("cuda:0")
    pipeline.transformer = streamed
    pipeline._gpu_runtime = None
    with patch("torch.cuda.mem_get_info", return_value=(1024**3, 1024**3)):
        pipeline.configure_gpu("cuda:0")
    assert pipeline._gpu_runtime.group_offload
    with torch.no_grad():
        actual = streamed(**arguments).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decoder_reservation_spills_only_required_blocks() -> None:
    pipeline = _pipeline("cuda:0")
    blocks = [MagicMock() for _ in range(4)]
    pipeline.transformer.transformer_blocks = blocks
    gib = 1024**3
    latents = torch.zeros(1, 16, 1, 128, 128)
    with (
        patch("torch.cuda.empty_cache"),
        patch(
            "torch.cuda.mem_get_info",
            side_effect=[(8 * gib, 32 * gib), (9 * gib, 32 * gib), (10 * gib, 32 * gib)],
        ),
    ):
        pipeline._reserve_decode_memory(latents)
    blocks[3].to.assert_called_once_with("cpu")
    blocks[2].to.assert_called_once_with("cpu")
    blocks[1].to.assert_not_called()
    blocks[0].to.assert_not_called()
    pipeline.transformer.to.assert_not_called()


def test_compaction_preserves_valid_suffix_after_middle_padding() -> None:
    pipeline = _pipeline()
    embeds = torch.arange(6).view(1, 6, 1, 1).expand(1, 6, 2, 4)
    mask = torch.tensor([[True, True, False, False, True, True]])
    with (
        patch.object(Krea2Pipeline, "get_text_hidden_states", return_value=(embeds, mask)),
        patch("torch.cuda.empty_cache"),
    ):
        compact, compact_mask = pipeline.get_text_hidden_states("fox", 6)
    assert compact[0, :, 0, 0].tolist() == [0, 1, 4, 5]
    assert compact_mask.all()


def test_compaction_keeps_columns_valid_for_any_prompt_in_batch() -> None:
    pipeline = _pipeline()
    embeds = torch.randn(2, 4, 2, 4)
    mask = torch.tensor([[True, False, False, True], [True, True, False, True]])
    with (
        patch.object(Krea2Pipeline, "get_text_hidden_states", return_value=(embeds, mask)),
        patch("torch.cuda.empty_cache"),
    ):
        compact, compact_mask = pipeline.get_text_hidden_states(["fox", "a fox"], 4)
    assert compact.shape[1] == 3
    assert compact_mask.tolist() == [[True, False, True], [True, True, True]]


def test_full_attention_mask_is_omitted_but_padding_mask_is_preserved() -> None:
    full = torch.ones(1, 4, dtype=torch.bool)
    _, kwargs = GpuKrea2Pipeline._omit_full_mask(None, (), {"encoder_attention_mask": full})
    assert kwargs["encoder_attention_mask"] is None
    partial = torch.tensor([[True, False, True]])
    _, kwargs = GpuKrea2Pipeline._omit_full_mask(None, (), {"encoder_attention_mask": partial})
    assert kwargs["encoder_attention_mask"] is partial


def test_padding_removal_preserves_transformer_image_conditioning() -> None:
    transformer = Krea2Transformer2DModel(
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        text_hidden_dim=32,
        num_text_layers=2,
        text_num_attention_heads=4,
        text_num_key_value_heads=4,
        text_intermediate_size=64,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(8, 4, 4),
    ).eval()
    mask = torch.tensor([[True, True, False, False, True, True]])
    text = torch.randn(1, 6, 2, 32)
    image = torch.randn(1, 8, 64)
    common = dict(hidden_states=image, timestep=torch.ones(1))
    with torch.no_grad():
        expected = transformer(
            **common,
            encoder_hidden_states=text,
            encoder_attention_mask=mask,
            position_ids=Krea2Pipeline.prepare_position_ids(6, 2, 4, torch.device("cpu")),
        ).sample
        actual = transformer(
            **common,
            encoder_hidden_states=text[:, mask[0]],
            encoder_attention_mask=None,
            position_ids=Krea2Pipeline.prepare_position_ids(4, 2, 4, torch.device("cpu")),
        ).sample
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
