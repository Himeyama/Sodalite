"""Embeds Sodalite's own generation-parameter metadata into PNG files."""

import json
from collections.abc import Mapping
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

METADATA_KEY = "sodalite_metadata"
PARAMETERS_KEY = "parameters"


def save_with_metadata(image: Image.Image, path: Path, parameters: Mapping[str, object]) -> None:
    """Save `image` as PNG at `path`, embedding `parameters` as a JSON tEXt chunk and a
    human-readable `parameters` tEXt chunk following the image-generation community's
    common plain-text layout, so other viewers/tools can display the generation settings."""
    png_info = PngInfo()
    png_info.add_text(METADATA_KEY, json.dumps(parameters, ensure_ascii=False))
    png_info.add_text(PARAMETERS_KEY, format_parameters_text(parameters))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, pnginfo=png_info)


def format_parameters_text(parameters: Mapping[str, object]) -> str:
    """Render `parameters` as the common human-readable text layout used across the
    image-generation community: prompt, then a negative-prompt line, then a settings line."""
    prompt = parameters.get("prompt", "")
    negative_prompt = parameters.get("negative_prompt", "")

    settings: list[str] = []
    if (steps := parameters.get("steps")) is not None:
        settings.append(f"Steps: {steps}")
    if (sampler := parameters.get("sampler")) is not None:
        settings.append(f"Sampler: {sampler}")
    if (cfg_scale := parameters.get("cfg_scale")) is not None:
        settings.append(f"CFG scale: {cfg_scale}")
    if (strength := parameters.get("strength")) is not None:
        settings.append(f"Denoising strength: {strength}")
    if (seed := parameters.get("seed")) is not None:
        settings.append(f"Seed: {seed}")
    if (width := parameters.get("width")) is not None and (
        height := parameters.get("height")
    ) is not None:
        settings.append(f"Size: {width}x{height}")
    if loras := parameters.get("loras"):
        lora_text = ", ".join(f"{lora['model_id']}:{lora['weight']}" for lora in loras)
        settings.append(f"LoRAs: {lora_text}")

    lines = [str(prompt), f"Negative prompt: {negative_prompt}", ", ".join(settings)]
    return "\n".join(lines)


def read_metadata(path: Path) -> dict[str, object] | None:
    """Read back the Sodalite metadata JSON embedded by `save_with_metadata`, if present."""
    with Image.open(path) as image:
        raw = image.text.get(METADATA_KEY) if hasattr(image, "text") else None
    return json.loads(raw) if raw else None
