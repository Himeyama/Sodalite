"""Small checkpoint fixtures for Krea 2 transformer loading."""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from sodalite_backend.inference.krea2 import load_krea2_text_encoder, load_krea2_transformer


def _tiny_transformer() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))


def test_load_scaled_fp8_checkpoint(tmp_path) -> None:
    path = tmp_path / "krea-fp8.safetensors"
    quantized = torch.tensor([[1, -2], [3, 4]], dtype=torch.float32).to(torch.float8_e4m3fn)
    save_file(
        {"0.weight": quantized, "0.weight_scale": torch.tensor(0.25)},
        path,
        metadata={
            "_quantization_metadata": json.dumps({"layers": {"0": {"format": "float8_e4m3fn"}}})
        },
    )

    with patch("sodalite_backend.inference.krea2.Krea2Transformer2DModel", _tiny_transformer):
        model = load_krea2_transformer(str(path))

    torch.testing.assert_close(
        model[0].weight, torch.tensor([[0.25, -0.5], [0.75, 1.0]], dtype=torch.bfloat16)
    )
    assert model[0].weight.dtype == torch.bfloat16


def test_reject_unscaled_fp8_checkpoint(tmp_path) -> None:
    path = tmp_path / "krea-unscaled.safetensors"
    save_file({"0.weight": torch.ones(2, 2).to(torch.float8_e4m3fn)}, path)

    with (
        patch("sodalite_backend.inference.krea2.Krea2Transformer2DModel", _tiny_transformer),
        pytest.raises(ValueError, match="missing its scale"),
    ):
        load_krea2_transformer(str(path))


def test_text_encoder_loads_checkpoint_architecture_before_extracting_model() -> None:
    wrapper = MagicMock()
    wrapper.model.eval.return_value.requires_grad_.return_value = wrapper.model
    info = {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}
    with patch("sodalite_backend.inference.krea2.Qwen3VLForConditionalGeneration") as model:
        model.from_pretrained.return_value = (wrapper, info)
        encoder = load_krea2_text_encoder()
    assert encoder is wrapper.model
    assert model.from_pretrained.call_args.kwargs["output_loading_info"] is True


@pytest.mark.parametrize("field", ["missing_keys", "mismatched_keys", "error_msgs"])
def test_text_encoder_rejects_random_or_incomplete_weights(field: str) -> None:
    info = {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}
    info[field] = ["model.language_model.layers.0.self_attn.q_proj.weight"]
    with patch("sodalite_backend.inference.krea2.Qwen3VLForConditionalGeneration") as model:
        model.from_pretrained.return_value = (MagicMock(), info)
        with pytest.raises(ValueError, match="Incomplete Krea 2 text encoder"):
            load_krea2_text_encoder()
