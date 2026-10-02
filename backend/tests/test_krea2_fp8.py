"""Check retained scaled FP8 against the original BF16 expansion and PEFT."""

import json
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import save_file

from sodalite_backend.inference.krea2 import load_krea2_transformer
from sodalite_backend.inference.krea2_fp8 import ScaledFP8Linear, retain_scaled_fp8


def _tiny_transformer() -> torch.nn.Module:
    model = torch.nn.Module()
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.to_q = torch.nn.Linear(2, 2, bias=False)
    model.transformer_blocks = torch.nn.ModuleList([block])
    return model


def test_retained_fp8_checkpoint_matches_bf16_expansion(tmp_path) -> None:
    checkpoint = tmp_path / "scaled.safetensors"
    quantized = torch.tensor([[1, -2], [3, 4]], dtype=torch.float32).to(torch.float8_e4m3fn)
    save_file(
        {"blocks.0.attn.wq.weight": quantized, "blocks.0.attn.wq.weight_scale": torch.tensor(0.25)},
        checkpoint,
        metadata={
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.attn.wq": {"format": "float8_e4m3fn"}}}
            )
        },
    )
    with patch("sodalite_backend.inference.krea2.Krea2Transformer2DModel", _tiny_transformer):
        retained = load_krea2_transformer(str(checkpoint), keep_scaled_fp8=True)
        expanded = load_krea2_transformer(str(checkpoint))
    layer = retained.transformer_blocks[0].attn.to_q
    assert isinstance(layer, ScaledFP8Linear)
    assert layer._parameters["weight"].dtype == torch.float8_e4m3fn
    assert layer.weight.dtype == torch.bfloat16
    assert layer._parameters["weight"].element_size() == 1
    inputs = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        layer(inputs), expanded.transformer_blocks[0].attn.to_q(inputs), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires real CUDA/ROCm")
def test_scaled_fp8_linear_and_lora_compute_on_gpu_without_quantizing_adapters() -> None:
    layer = torch.nn.Linear(2, 2, bias=False)
    layer.weight = torch.nn.Parameter(
        torch.tensor([[1, -2], [3, 4]], dtype=torch.float32).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    retain_scaled_fp8(layer, torch.tensor(0.25))
    layer.to("cuda")
    inputs = torch.tensor([[1, 2]], device="cuda", dtype=torch.bfloat16)
    expected_base = torch.nn.functional.linear(inputs, layer.weight)
    torch.testing.assert_close(layer(inputs), expected_base, rtol=0, atol=0)
    lora = LoraLinear(
        layer, "test", config=LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0), r=2, lora_alpha=2
    )
    assert lora.lora_A["test"].weight.dtype == torch.bfloat16
    assert lora.lora_B["test"].weight.dtype == torch.bfloat16
    with torch.no_grad():
        lora.lora_A["test"].weight.fill_(0.5)
        lora.lora_B["test"].weight.fill_(0.25)
    expected = expected_base + lora.lora_B["test"](lora.lora_A["test"](inputs))
    torch.testing.assert_close(lora(inputs), expected, rtol=0, atol=0)
    assert layer._parameters["weight"].dtype == torch.float8_e4m3fn
