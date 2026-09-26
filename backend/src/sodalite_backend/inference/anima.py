"""Load original-format Anima transformer checkpoints with the shared Diffusers components."""

from collections.abc import Callable
from pathlib import Path

import torch
from diffusers import ComponentsManager, CosmosTransformer3DModel, ModularPipeline
from huggingface_hub import constants
from safetensors import SafetensorError, safe_open
from transformers import Qwen2TokenizerFast, T5TokenizerFast

ANIMA_DIFFUSERS_MODEL_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"
_ANIMA_PREFIXES = ("net.", "model.diffusion_model.", "diffusion_model.", "")
_ANIMA_MARKERS = ("x_embedder.proj.1.weight", "llm_adapter.embed.weight")


def is_anima_checkpoint(path: str) -> bool:
    """Identify Anima's Cosmos transformer weights from their tensor names."""
    if Path(path).suffix.lower() != ".safetensors":
        return False

    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
    except (OSError, SafetensorError):
        return False

    return any(
        all(f"{prefix}{marker}" in keys for marker in _ANIMA_MARKERS)
        for prefix in _ANIMA_PREFIXES
    )


def _load_transformer(path: str, dtype: torch.dtype) -> CosmosTransformer3DModel:
    """Load the selected Anima transformer, normalizing ComfyUI's outer model prefix if present."""
    comfy_prefixes = ("model.diffusion_model.", "diffusion_model.")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = checkpoint.keys()
        prefix = next(
            (candidate for candidate in comfy_prefixes if any(key.startswith(candidate) for key in keys)),
            None,
        )
        if prefix is None:
            checkpoint_source: str | dict[str, torch.Tensor] = path
        else:
            checkpoint_source = {
                key.removeprefix(prefix): checkpoint.get_tensor(key)
                for key in keys
                if key.startswith(prefix)
            }

    return CosmosTransformer3DModel.from_single_file(
        checkpoint_source,
        config=ANIMA_DIFFUSERS_MODEL_ID,
        subfolder="transformer",
        torch_dtype=dtype,
    )


def _download_location() -> tuple[str, str]:
    source = f"https://huggingface.co/{ANIMA_DIFFUSERS_MODEL_ID}"
    destination = str(
        Path(constants.HF_HUB_CACHE)
        / f"models--{ANIMA_DIFFUSERS_MODEL_ID.replace('/', '--')}"
    )
    return source, destination


def load_anima_pipeline(
    checkpoint_path: str,
    device: torch.device | str,
    dtype: torch.dtype,
    on_stage: Callable[[str, str | None, str | None], None] | None = None,
) -> ModularPipeline:
    """Use a local transformer checkpoint with Anima's official text and image components."""
    components_manager = None
    if str(device) == "cuda":
        components_manager = ComponentsManager()
        components_manager.enable_auto_cpu_offload(device=device)

    def stage(message: str, downloading: bool = False) -> None:
        if on_stage is None:
            return
        source, destination = _download_location() if downloading else (None, None)
        on_stage(message, source, destination)

    stage("ANIMA のパイプライン設定を読み込み中", downloading=True)
    pipeline = ModularPipeline.from_pretrained(
        ANIMA_DIFFUSERS_MODEL_ID,
        components_manager=components_manager,
    )

    stage("ANIMA のチェックポイントを読み込み中")
    transformer = _load_transformer(checkpoint_path, dtype)
    pipeline.update_components(transformer=transformer)

    # The published model only includes tokenizer.json files. Its modular index
    # names the slow tokenizer classes, which need files absent from the repo.
    stage("ANIMA のトークナイザーをダウンロード・読み込み中", downloading=True)
    pipeline.update_components(
        tokenizer=Qwen2TokenizerFast.from_pretrained(
            ANIMA_DIFFUSERS_MODEL_ID, subfolder="tokenizer"
        ),
        t5_tokenizer=T5TokenizerFast.from_pretrained(
            ANIMA_DIFFUSERS_MODEL_ID,
            subfolder="t5_tokenizer",
            # The published config stores this as a list, but Transformers 4
            # expects a mapping. T5TokenizerFast adds its 100 sentinel tokens.
            extra_special_tokens={},
        ),
    )

    stage("ANIMA のテキストエンコーダー・VAEをダウンロード・読み込み中", downloading=True)
    pipeline.load_components(torch_dtype=dtype)

    missing_components = [
        name
        for name in (
            "text_encoder",
            "tokenizer",
            "t5_tokenizer",
            "text_conditioner",
            "guider",
            "scheduler",
            "vae",
            "image_processor",
            "transformer",
        )
        if getattr(pipeline, name, None) is None
    ]
    if missing_components:
        raise RuntimeError(
            "Failed to load required Anima components: " + ", ".join(missing_components)
        )

    stage("ANIMA パイプラインを準備中")
    if components_manager is None:
        pipeline.to(device)
    return pipeline
