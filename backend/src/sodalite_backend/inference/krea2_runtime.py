"""Keep Krea 2 computation on CUDA/ROCm while avoiding redundant model transfers."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers import Krea2Pipeline
from diffusers.models.autoencoders.vae import DecoderOutput

logger = logging.getLogger(__name__)
_GIB = 1024**3


@dataclass
class _GpuRuntime:
    device: torch.device
    decode: Callable[..., DecoderOutput | tuple[torch.Tensor]]
    group_offload: bool = False
    cache_key: tuple[tuple[str, ...], int] | None = None
    cache: tuple[torch.Tensor, torch.Tensor] | None = None
    offload_for_decode: bool = False


class GpuKrea2Pipeline(Krea2Pipeline):
    """Cache one prompt and retain the denoiser between images on roomy GPUs.

    CPU holds inactive weights only. Text encoding, denoising and VAE decoding
    execute on the selected GPU. Small GPUs stream two transformer blocks at a
    time instead of attempting to place the entire 26 GB transformer in VRAM.
    The constructor is inherited to preserve Diffusers' component discovery.
    """

    _gpu_runtime: _GpuRuntime | None = None

    def configure_gpu(self, device: torch.device | str) -> None:
        """Select residency or block offload according to actual free VRAM."""
        target = torch.device(device)
        if target.index is None:
            target = torch.device(target.type, torch.cuda.current_device())
        if self._gpu_runtime is not None:
            if self._gpu_runtime.device != target:
                raise ValueError("Create a new Krea 2 pipeline to use a different GPU.")
            return
        self._gpu_runtime = _GpuRuntime(device=target, decode=self.vae.decode)
        weight_bytes = sum(t.numel() * t.element_size() for t in self.transformer.parameters())
        free_bytes, _ = torch.cuda.mem_get_info(target)
        # Leave space for decoder workspace and allocator fragmentation. On
        # Windows ROCm an overfull reservation silently spills into shared RAM.
        if weight_bytes + 12 * _GIB > free_bytes:
            self.transformer.enable_group_offload(
                onload_device=target,
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=2,
                use_stream=False,
            )
            self._gpu_runtime.group_offload = True
            logger.info("Krea 2 uses GPU block offload (free VRAM %.1f GiB)", free_bytes / _GIB)
        else:
            logger.info(
                "Krea 2 keeps its transformer on GPU (free VRAM %.1f GiB)", free_bytes / _GIB
            )
        self.vae.decode = self._decode_on_gpu
        self.transformer.register_forward_pre_hook(self._omit_full_mask, with_kwargs=True)

    @staticmethod
    def _omit_full_mask(
        _module: torch.nn.Module, args: tuple, kwargs: dict[str, object]
    ) -> tuple[tuple, dict[str, object]]:
        mask = kwargs.get("encoder_attention_mask")
        if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and bool(mask.all()):
            # Once padding has been removed, an all-valid mask is redundant.
            # ROCm's flash kernel rejects non-null masks, even all-valid ones.
            kwargs["encoder_attention_mask"] = None
        return args, kwargs

    @property
    def _execution_device(self) -> torch.device:
        if self._gpu_runtime is not None:
            return self._gpu_runtime.device
        return super()._execution_device

    @torch.no_grad()
    def get_text_hidden_states(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runtime = self._gpu_runtime
        if runtime is None:
            return super().get_text_hidden_states(prompt, max_sequence_length, device)
        key = ((prompt,) if isinstance(prompt, str) else tuple(prompt), max_sequence_length)
        target = device or runtime.device
        if runtime.cache_key == key and runtime.cache is not None:
            return tuple(t.to(target) for t in runtime.cache)

        # A new prompt needs the large Qwen encoder. Free the denoiser first,
        # then discard the encoder's GPU weights immediately after encoding.
        runtime.cache = None
        runtime.cache_key = None
        if not runtime.group_offload:
            self.transformer.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        try:
            self.text_encoder.to(target)
            embeds, mask = super().get_text_hidden_states(prompt, max_sequence_length, target)
        finally:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        # Padding columns cannot influence valid text/image tokens, and all
        # transformer text rotary coordinates are zero. Remove only columns
        # that are invalid for every prompt; preserve the suffix after padding.
        valid_columns = mask.any(dim=0)
        embeds = embeds[:, valid_columns].contiguous()
        mask = mask[:, valid_columns].contiguous()
        runtime.cache = (embeds.detach(), mask.detach())
        runtime.cache_key = key
        return embeds, mask

    def encode_prompt(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        embeds, mask = super().encode_prompt(*args, **kwargs)
        runtime = self._gpu_runtime
        if runtime is not None and not runtime.group_offload:
            self.transformer.to(runtime.device)
        return embeds, mask

    def _decode_on_gpu(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | tuple[torch.Tensor]:
        runtime = self._gpu_runtime
        if runtime.offload_for_decode and not runtime.group_offload:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
        elif not runtime.group_offload:
            self._reserve_decode_memory(z)
        self.vae.to(runtime.device)
        try:
            return runtime.decode(z, return_dict=return_dict)
        except torch.cuda.OutOfMemoryError:
            if runtime.group_offload or runtime.offload_for_decode:
                raise
            # Retry only the decoder, with the exact same latents. Preserve
            # full-image decoding and BF16 weights instead of changing quality.
            logger.warning("Krea 2 VAE needs more VRAM; releasing transformer before GPU decoding")
            runtime.offload_for_decode = True
        self.transformer.to("cpu")
        torch.cuda.empty_cache()
        return runtime.decode(z, return_dict=return_dict)

    def _reserve_decode_memory(self, z: torch.Tensor) -> None:
        # Windows ROCm can oversubscribe VRAM into shared RAM without raising
        # OOM. Reserve decoder workspace proactively, rather than relying on
        # exceptions. Measured full-frame BF16 decode needs ~9 GiB at 1024².
        runtime = self._gpu_runtime
        if runtime.device.type != "cuda" or z.ndim != 5:
            return
        reserve = max(2 * _GIB, z.shape[0] * z.shape[-2] * z.shape[-1] * 10 * _GIB // 128**2)
        torch.cuda.empty_cache()
        for block in reversed(self.transformer.transformer_blocks):
            free_bytes, _ = torch.cuda.mem_get_info(runtime.device)
            if free_bytes >= reserve:
                break
            block.to("cpu")
            torch.cuda.empty_cache()

    def maybe_free_model_hooks(self) -> None:
        """Release inactive GPU weights, keeping the denoiser and cached prompt."""
        if self._gpu_runtime is not None:
            self.vae.to("cpu")
            self.text_encoder.to("cpu")
        super().maybe_free_model_hooks()
        if self._gpu_runtime is not None:
            # Return the decoder's temporary allocations to the driver rather
            # than retaining a near-full VRAM reservation between images.
            torch.cuda.empty_cache()

    def release_gpu(self) -> None:
        """Release residency and cached tensors before switching models or after errors."""
        if self._gpu_runtime is None:
            return
        self._gpu_runtime.cache = None
        self._gpu_runtime.cache_key = None
        self.transformer.to("cpu")
        self.vae.to("cpu")
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()
