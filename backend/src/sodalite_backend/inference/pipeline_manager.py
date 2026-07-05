"""Loads and holds the active diffusers pipeline for inference."""

import gc
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from sodalite_backend.inference.samplers import SAMPLER_CLASSES
from sodalite_backend.schemas.generation import LoraSpec, Sampler


class PipelineManager:
    """Owns a single loaded text-to-image pipeline, moved onto the best available device."""

    def __init__(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self._pipeline = self._load_pipeline(model_id)

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
        self.model_id = model_id

        del old_pipeline
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _load_pipeline(self, model_id: str) -> DiffusionPipeline:
        dtype = torch.float16 if self.device == "cuda" else torch.float32

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

    def set_sampler(self, sampler: Sampler) -> None:
        scheduler_cls = SAMPLER_CLASSES[sampler]
        self._pipeline.scheduler = scheduler_cls.from_config(self._pipeline.scheduler.config)

    def _apply_loras(self, loras: list[LoraSpec]) -> None:
        """Load the requested LoRAs onto the pipeline and activate them by weight.

        Each LoRA is loaded under a distinct adapter name so multiple can be
        blended in one generation. The base checkpoint and the LoRA must share
        the same architecture (SD1.5 vs SDXL); a mismatch surfaces as a load
        error from diffusers.
        """
        adapter_names: list[str] = []
        adapter_weights: list[float] = []
        for index, lora in enumerate(loras):
            adapter_name = f"lora_{index}"
            self._load_single_lora(lora.model_id, adapter_name)
            adapter_names.append(adapter_name)
            adapter_weights.append(lora.weight)

        self._pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

    def _load_single_lora(self, model_id: str, adapter_name: str) -> None:
        """Load one LoRA, tolerating text-encoder sub-weights diffusers can't parse.

        The convenient `load_lora_weights` loads the UNet and both text encoders
        in one call, but diffusers 0.39 raises `IndexError` while inferring the
        rank of some community LoRAs whose text-encoder keys it fails to match
        (e.g. LoRAs trained only against one of SDXL's two encoders). That aborts
        the whole load and surfaces as a 500. We instead drive the same pipeline
        loaders directly so the UNet and each text encoder are loaded
        independently, skipping only the text-encoder part diffusers chokes on.
        """
        pipeline = self._pipeline
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

    def _clear_loras(self) -> None:
        """Remove any LoRA weights so they don't leak into later generations or model switches."""
        self._pipeline.unload_lora_weights()

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
        loras: list[LoraSpec] | None = None,
    ):
        self.set_sampler(sampler)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if loras:
            self._apply_loras(loras)
        try:
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator,
            )
        finally:
            if loras:
                self._clear_loras()
        return result.images[0]
