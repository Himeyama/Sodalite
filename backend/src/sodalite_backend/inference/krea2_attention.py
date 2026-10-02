"""Use fused ROCm SDPA for Krea's grouped-query attention without changing weights."""

import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_krea2 import Krea2Attention, Krea2AttnProcessor
from torch.nn import functional as F


class RocmKrea2AttnProcessor(Krea2AttnProcessor):
    """Expand KV heads so ROCm can use memory-efficient attention with a mask.

    PyTorch 2.9 ROCm rejects unequal head counts for fused SDPA, while its flash
    kernel rejects Krea's padding mask. Explicitly repeating the smaller KV
    tensors implements the same GQA mapping and avoids the quadratic-memory
    math fallback. This keeps all model weights and computation in BF16/FP32.
    """

    def __call__(
        self,
        attn: Krea2Attention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.norm_q(
            attn.to_q(hidden_states).reshape(
                *hidden_states.shape[:2], attn.num_heads, attn.head_dim
            )
        )
        key = attn.norm_k(
            attn.to_k(hidden_states).reshape(
                *hidden_states.shape[:2], attn.num_kv_heads, attn.head_dim
            )
        )
        value = attn.to_v(hidden_states).reshape(
            *hidden_states.shape[:2], attn.num_kv_heads, attn.head_dim
        )
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        repeats = attn.num_heads // attn.num_kv_heads
        key = key.repeat_interleave(repeats, dim=2)
        value = value.repeat_interleave(repeats, dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).flatten(2)
        return attn.to_out[0](attended * torch.sigmoid(attn.to_gate(hidden_states)))


def enable_rocm_krea2_attention(transformer: torch.nn.Module) -> None:
    """Use the ROCm processor only on Krea attention layers with grouped heads."""
    for module in transformer.modules():
        if isinstance(module, Krea2Attention) and module.num_heads != module.num_kv_heads:
            module.set_processor(RocmKrea2AttnProcessor())
