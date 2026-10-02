"""Check GQA equivalence, masks, rotary embeddings and real ROCm fused dispatch."""

import copy

import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    Krea2AttnProcessor,
    Krea2RotaryPosEmbed,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

from sodalite_backend.inference.krea2_attention import (
    RocmKrea2AttnProcessor,
    enable_rocm_krea2_attention,
)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rotary", [False, True])
def test_expanded_heads_match_original_gqa(masked: bool, rotary: bool) -> None:
    torch.manual_seed(12)
    original = Krea2Attention(hidden_size=128, num_heads=8, num_kv_heads=2).eval()
    optimized = copy.deepcopy(original)
    optimized.set_processor(RocmKrea2AttnProcessor())
    inputs = torch.randn(2, 16, 128)
    mask = torch.ones(2, 1, 1, 16, dtype=torch.bool) if masked else None
    if mask is not None:
        mask[:, :, :, -3:] = False
    rope = (
        Krea2RotaryPosEmbed(1000, [8, 4, 4])(torch.arange(16).view(16, 1).expand(16, 3))
        if rotary
        else None
    )
    with torch.no_grad():
        expected = original(inputs, attention_mask=mask, image_rotary_emb=rope)
        actual = optimized(inputs, attention_mask=mask, image_rotary_emb=rope)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_only_grouped_krea_attention_is_changed() -> None:
    grouped = Krea2Attention(hidden_size=128, num_heads=8, num_kv_heads=2)
    equal_heads = Krea2Attention(hidden_size=128, num_heads=8, num_kv_heads=8)
    transformer = torch.nn.ModuleList([grouped, equal_heads])
    enable_rocm_krea2_attention(transformer)
    assert isinstance(grouped.processor, RocmKrea2AttnProcessor)
    assert type(equal_heads.processor) is Krea2AttnProcessor


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None, reason="Requires real ROCm"
)
def test_rocm_fused_kernel_with_padding_mask_matches_math_gqa() -> None:
    torch.manual_seed(12)
    original = (
        Krea2Attention(hidden_size=768, num_heads=48, num_kv_heads=12)
        .to("cuda", dtype=torch.bfloat16)
        .eval()
    )
    optimized = copy.deepcopy(original)
    optimized.set_processor(RocmKrea2AttnProcessor())
    inputs = torch.randn(1, 32, 768, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 1, 32, device="cuda", dtype=torch.bool)
    mask[..., -4:] = False
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected = original(inputs, attention_mask=mask)
    # Disallow math and flash: passing proves efficient attention really ran.
    with torch.no_grad(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = optimized(inputs, attention_mask=mask)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
