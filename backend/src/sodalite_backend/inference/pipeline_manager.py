"""Loads and holds the active diffusers pipeline for inference."""

import contextlib
import gc
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

from sodalite_backend.inference.samplers import SAMPLER_CLASSES
from sodalite_backend.inference.prompt_weights import (
    apply_attention_weights,
    has_attention_syntax,
    parse_prompt_attention,
)
from sodalite_backend.schemas.generation import LoraSpec, Sampler


class ModelNotReadyError(Exception):
    """Raised when an operation needs the pipeline but no model has finished loading yet."""


def _get_directml_device() -> torch.device | None:
    """Return the DirectML device when its optional Windows backend is installed."""
    try:
        import torch_directml
    except ImportError:
        return None

    try:
        return torch_directml.device()
    except RuntimeError:
        # A package may remain installed after the machine changes, or no
        # DirectX 12 adapter may be available. In either case CPU is safer.
        return None


def _select_device() -> tuple[torch.device | str, str]:
    """Select a native CUDA/HIP device first, then DirectML, and finally CPU."""
    if torch.cuda.is_available():
        # PyTorch deliberately exposes ROCm devices through its torch.cuda API.
        # torch.version.hip is the supported way to distinguish that build from CUDA.
        backend = "rocm" if torch.version.hip is not None else "cuda"
        return "cuda", backend

    if directml_device := _get_directml_device():
        return directml_device, "directml"

    return "cpu", "cpu"


class PipelineManager:
    """Owns a single loaded text-to-image pipeline, moved onto the best available device.

    The initial model load is kicked off separately from construction (see
    `load_initial_model`) so the server can start accepting requests (health
    checks, gallery browsing) while the first model is still loading onto the
    device. Any operation that needs the pipeline raises `ModelNotReadyError`
    until that load finishes.
    """

    def __init__(self) -> None:
        self.device, self.device_backend = _select_device()
        self.model_id: str | None = None
        self._pipeline: DiffusionPipeline | None = None
        self._image_to_image_pipeline: DiffusionPipeline | None = None

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    def load_initial_model(self, model_id: str) -> None:
        """Load the first model. Intended to run once, before any `load_model` call."""
        self._pipeline = self._load_pipeline(model_id)
        self._image_to_image_pipeline = None
        self.model_id = model_id

    def load_model(self, model_id: str) -> None:
        """Replace the currently loaded pipeline with a different model.

        The new pipeline is loaded before the old one is dropped, so a failed load
        leaves the previously active model intact. The old pipeline is then released
        explicitly to free the device memory it held, which on CUDA would otherwise
        stay reserved and make the next load run out of VRAM.
        """
        new_pipeline = self._load_pipeline(model_id)

        old_pipeline = self._pipeline
        self._pipeline = new_pipeline
        self._image_to_image_pipeline = None
        self.model_id = model_id

        del old_pipeline
        gc.collect()
        if self.device_backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()

    def _require_pipeline(self) -> DiffusionPipeline:
        if self._pipeline is None:
            raise ModelNotReadyError("No model has finished loading yet.")
        return self._pipeline

    def _load_pipeline(self, model_id: str) -> DiffusionPipeline:
        dtype = (
            torch.float16
            if self.device_backend in {"cuda", "rocm", "directml"}
            else torch.float32
        )

        if Path(model_id).is_file():
            return self._load_single_file_pipeline(model_id, dtype)

        return AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype).to(
            self.device
        )

    def _load_single_file_pipeline(self, model_path: str, dtype: torch.dtype) -> DiffusionPipeline:
        try:
            return StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype).to(
                self.device
            )
        except Exception:
            return StableDiffusionPipeline.from_single_file(
                model_path, torch_dtype=dtype, safety_checker=None
            ).to(self.device)

    def set_sampler(self, sampler: Sampler, pipeline: DiffusionPipeline | None = None) -> None:
        pipeline = pipeline or self._require_pipeline()
        scheduler_cls = SAMPLER_CLASSES[sampler]
        pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)

    def _apply_loras(self, loras: list[LoraSpec]) -> None:
        """Load the requested LoRAs onto the pipeline and activate them by weight.

        Each LoRA is loaded under a distinct adapter name so multiple can be
        blended in one generation. A LoRA whose architecture doesn't match the
        active checkpoint (e.g. an SDXL LoRA left selected after switching to an
        SD1.5 model) fails to load; that one is skipped so the rest still apply
        and generation isn't aborted.
        """
        adapter_names: list[str] = []
        adapter_weights: list[float] = []
        for index, lora in enumerate(loras):
            adapter_name = f"lora_{index}"
            if self._load_single_lora(lora.model_id, adapter_name):
                adapter_names.append(adapter_name)
                adapter_weights.append(lora.weight)

        if adapter_names:
            self._require_pipeline().set_adapters(adapter_names, adapter_weights=adapter_weights)

    def _load_single_lora(self, model_id: str, adapter_name: str) -> bool:
        """Load one LoRA onto the pipeline, returning whether it was applied.

        The convenient `load_lora_weights` loads the UNet and both text encoders
        in one call, but diffusers 0.39 raises `IndexError` while inferring the
        rank of some community LoRAs whose text-encoder keys it fails to match
        (e.g. LoRAs trained only against one of SDXL's two encoders). That aborts
        the whole load and surfaces as a 500. We instead drive the same pipeline
        loaders directly so the UNet and each text encoder are loaded
        independently, skipping only the text-encoder part diffusers chokes on.

        Returns False (and unloads any partial state) when the LoRA is
        incompatible with the active checkpoint, so the caller can move on.
        """
        pipeline = self._require_pipeline()
        try:
            state_dict, network_alphas, metadata = pipeline.lora_state_dict(
                model_id, unet_config=pipeline.unet.config, return_lora_metadata=True
            )
            pipeline.load_lora_into_unet(
                state_dict,
                network_alphas=network_alphas,
                unet=pipeline.unet,
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=pipeline,
            )
        except (ValueError, RuntimeError, KeyError):
            # Architecture mismatch (SD1.5 vs SDXL) or an unreadable checkpoint;
            # drop just this half-registered adapter (leaving any already-applied
            # LoRAs intact) and skip it.
            with contextlib.suppress(ValueError, KeyError):
                pipeline.delete_adapters(adapter_name)
            return False

        text_encoders = [(pipeline.text_encoder, "text_encoder")]
        if getattr(pipeline, "text_encoder_2", None) is not None:
            text_encoders.append((pipeline.text_encoder_2, "text_encoder_2"))

        for text_encoder, prefix in text_encoders:
            try:
                pipeline.load_lora_into_text_encoder(
                    state_dict,
                    network_alphas=network_alphas,
                    text_encoder=text_encoder,
                    prefix=prefix,
                    lora_scale=pipeline.lora_scale,
                    adapter_name=adapter_name,
                    metadata=metadata,
                    _pipeline=pipeline,
                )
            except IndexError:
                # This LoRA carries no parseable weights for this text encoder;
                # the UNet (and any other encoder) still applies, so skip it.
                continue

        return True

    def _clear_loras(self) -> None:
        """Remove any LoRA weights so they don't leak into later generations or model switches."""
        self._require_pipeline().unload_lora_weights()

    def _get_image_to_image_pipeline(self) -> DiffusionPipeline:
        """Adapt the loaded model without a second checkpoint load or VRAM copy."""
        if self._image_to_image_pipeline is None:
            self._image_to_image_pipeline = AutoPipelineForImage2Image.from_pipe(
                self._require_pipeline()
            )
            if self.device_backend == "rocm":
                # Reduces SDXL decoder peak VRAM on Radeon without moving the
                # model or tensors off the HIP device.
                self._image_to_image_pipeline.enable_vae_slicing()
        return self._image_to_image_pipeline

    def _weighted_prompt_arguments(
        self, pipeline: DiffusionPipeline, prompt: str, negative_prompt: str, cfg_scale: float
    ) -> dict[str, object]:
        """Encode prompt-attention syntax once, then pass weighted embeds to diffusers."""
        clean_prompt = "".join(fragment.text for fragment in parse_prompt_attention(prompt))
        clean_negative_prompt = "".join(
            fragment.text for fragment in parse_prompt_attention(negative_prompt)
        )
        use_guidance = cfg_scale > 1.0

        if getattr(pipeline, "text_encoder_2", None) is not None:
            prompt_embeds, negative_embeds, pooled_embeds, negative_pooled_embeds = (
                pipeline.encode_prompt(
                    prompt=clean_prompt,
                    prompt_2=None,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=use_guidance,
                    negative_prompt=clean_negative_prompt or None,
                    negative_prompt_2=None,
                )
            )
            arguments: dict[str, object] = {
                "prompt_embeds": apply_attention_weights(
                    prompt_embeds, pipeline.tokenizer, parse_prompt_attention(prompt)
                ),
                "pooled_prompt_embeds": pooled_embeds,
            }
            if negative_embeds is not None and negative_pooled_embeds is not None:
                arguments["negative_prompt_embeds"] = apply_attention_weights(
                    negative_embeds, pipeline.tokenizer, parse_prompt_attention(negative_prompt)
                )
                arguments["negative_pooled_prompt_embeds"] = negative_pooled_embeds
            return arguments

        prompt_embeds, negative_embeds = pipeline.encode_prompt(
            prompt=clean_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=use_guidance,
            negative_prompt=clean_negative_prompt or None,
        )
        arguments = {
            "prompt_embeds": apply_attention_weights(
                prompt_embeds, pipeline.tokenizer, parse_prompt_attention(prompt)
            )
        }
        if negative_embeds is not None:
            arguments["negative_prompt_embeds"] = apply_attention_weights(
                negative_embeds, pipeline.tokenizer, parse_prompt_attention(negative_prompt)
            )
        return arguments

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
        sampler: Sampler,
        seed: int | None,
        batch_size: int = 1,
        loras: list[LoraSpec] | None = None,
        initial_image: Image.Image | None = None,
        strength: float = 0.4,
        should_stop: Callable[[], bool] | None = None,
        on_step: Callable[[int, int], None] | None = None,
    ) -> Iterator[Image.Image]:
        """Yield `batch_size` images one at a time, each as soon as it's ready.

        Generating one-by-one (rather than diffusers' native batching) keeps peak
        VRAM usage flat regardless of batch size, lets each image use a distinct,
        reproducible seed (base seed + index) instead of all sharing one, and lets
        the caller react to (save, display) each image as it finishes instead of
        waiting for the whole batch. `should_stop` is polled between images so a
        long batch can be cancelled without waiting for it to run to completion.
        `on_step`, if given, is called after each denoising step with
        `(step_index, total_steps)` so the caller can surface within-image progress.

        Raises `ModelNotReadyError` if no model has finished loading yet.
        """
        pipeline = (
            self._get_image_to_image_pipeline() if initial_image is not None else self._require_pipeline()
        )
        self.set_sampler(sampler, pipeline)

        if loras:
            self._apply_loras(loras)
        try:
            for index in range(batch_size):
                if should_stop is not None and should_stop():
                    return

                generator = None
                if seed is not None:
                    # DirectML does not implement a private-use-device Generator.
                    # diffusers accepts a CPU generator while tensors run on DML.
                    generator_device = "cpu" if self.device_backend == "directml" else self.device
                    generator = torch.Generator(device=generator_device).manual_seed(seed + index)

                def report_step(
                    _pipeline: DiffusionPipeline,
                    step_index: int,
                    _timestep: int,
                    callback_kwargs: dict[str, object],
                ) -> dict[str, object]:
                    if on_step is not None:
                        on_step(step_index + 1, steps)
                    return callback_kwargs

                arguments: dict[str, object] = dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=report_step if on_step is not None else None,
                )
                if has_attention_syntax(prompt) or has_attention_syntax(negative_prompt):
                    arguments.pop("prompt")
                    arguments.pop("negative_prompt")
                    arguments.update(
                        self._weighted_prompt_arguments(pipeline, prompt, negative_prompt, cfg_scale)
                    )
                if initial_image is not None:
                    arguments["image"] = initial_image
                    arguments["strength"] = strength
                result = pipeline(**arguments)
                yield result.images[0]
        finally:
            if loras:
                self._clear_loras()
