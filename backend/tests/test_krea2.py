"""Small checkpoint fixtures for Krea 2 transformer loading."""

import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sodalite_backend.inference.krea2 import load_krea2_transformer


def _tiny_transformer() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))


def test_load_scaled_fp8_checkpoint(tmp_path) -> None:
    path = tmp_path / "krea-fp8.safetensors"
    quantized = torch.tensor([[1, -2], [3, 4]], dtype=torch.float32).to(torch.float8_e4m3fn)
    save_file(
        {"0.weight": quantized, "0.weight_scale": torch.tensor(0.25)},
        path,
        metadata={"_quantization_metadata": json.dumps({"layers": {"0": {"format": "float8_e4m3fn"}}})},
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

    with patch("sodalite_backend.inference.krea2.Krea2Transformer2DModel", _tiny_transformer):
        with pytest.raises(ValueError, match="missing its scale"):
            load_krea2_transformer(str(path))
