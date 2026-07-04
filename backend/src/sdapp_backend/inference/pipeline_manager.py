"""Loads and holds the active diffusers pipeline for inference."""

from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from sdapp_backend.inference.samplers import SAMPLER_CLASSES
from sdapp_backend.schemas.generation import Sampler


class PipelineManager:
    """Owns a single loaded text-to-image pipeline, moved onto the best available device."""

    def __init__(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self._pipeline = self._load_pipeline(model_id)

    def load_model(self, model_id: str) -> None:
        """Replace the currently loaded pipeline with a different model."""
        self._pipeline = self._load_pipeline(model_id)
        self.model_id = model_id

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
    ):
        self.set_sampler(sampler)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0]
