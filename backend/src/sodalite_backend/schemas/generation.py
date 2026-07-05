"""Request/response schemas for the generation API."""

from typing import Literal

from pydantic import BaseModel, Field

Sampler = Literal["euler_a", "euler", "dpmpp_2m", "ddim", "lms"]

JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class LoraSpec(BaseModel):
    """A LoRA to apply during generation, identified the same way as a base model.

    `model_id` is either a Hugging Face repo id or an absolute path to a
    single-file `.safetensors` LoRA. `weight` scales its influence; negative
    values invert the effect.
    """

    model_id: str
    weight: float = Field(default=1.0, ge=-2.0, le=2.0)


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=64, le=2048, multiple_of=8)
    height: int = Field(default=512, ge=64, le=2048, multiple_of=8)
    sampler: Sampler = "euler_a"
    seed: int | None = None
    loras: list[LoraSpec] = Field(default_factory=list)


class GenerationJob(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    image_url: str | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    is_active: bool
    size_on_disk_bytes: int


class SetActiveModelRequest(BaseModel):
    model_id: str


class ImportModelRequest(BaseModel):
    model_path: str


class LoraFileInfo(BaseModel):
    """A LoRA file available locally, identified by its absolute path."""

    lora_id: str
    size_on_disk_bytes: int


class ImportLoraRequest(BaseModel):
    lora_path: str
