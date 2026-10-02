"""Retain existing scaled-FP8 weights while computing each linear layer in BF16."""

import torch
from torch.nn import functional as F


class ScaledFP8Linear(torch.nn.Linear):
    """Dequantize a checkpoint's scaled weight on the GPU only when it is needed.

    This does not quantize BF16 checkpoints or require native FP8 GEMM support.
    The FP32 scaling then BF16 rounding matches the original loader exactly.
    Only the current linear layer needs a temporary expanded weight.
    """

    @property
    def weight(self) -> torch.Tensor:
        # PEFT reads this logical dtype when creating/loading LoRA matrices.
        # Keep the actual registered Parameter in FP8 for storage and transfer.
        stored = self._parameters["weight"]
        return stored.to(torch.float32).mul_(self._krea2_weight_scale).to(torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight.to(inputs.dtype), self.bias)


def retain_scaled_fp8(module: torch.nn.Linear, scale: torch.Tensor) -> None:
    """Keep the existing Linear parameter names so Diffusers and PEFT can find them."""
    module.weight.requires_grad_(False)
    module.__class__ = ScaledFP8Linear
    module.register_buffer("_krea2_weight_scale", scale.clone())
