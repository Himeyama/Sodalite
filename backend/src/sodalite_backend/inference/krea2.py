"""Load an original-format Krea 2 Turbo transformer with public Qwen components."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)
from huggingface_hub import constants, try_to_load_from_cache
from safetensors import SafetensorError, safe_open
from transformers import AutoTokenizer, Qwen3VLModel

TEXT_ENCODER_ID = "Qwen/Qwen3-VL-4B-Instruct"
VAE_ID = "Qwen/Qwen-Image"


def _cached(repo_id: str, filename: str) -> bool:
    return isinstance(try_to_load_from_cache(repo_id, filename), str)


def _text_encoder_needs_download() -> bool:
    index_path = try_to_load_from_cache(TEXT_ENCODER_ID, "model.safetensors.index.json")
    if not isinstance(index_path, str):
        return True
    with open(index_path, encoding="utf-8") as index_file:
        shards = set(json.load(index_file)["weight_map"].values())
    return not all(_cached(TEXT_ENCODER_ID, name) for name in shards | {"tokenizer.json", "tokenizer_config.json"})


def _vae_needs_download() -> bool:
    return not all(
        _cached(VAE_ID, name)
        for name in ("vae/config.json", "vae/diffusion_pytorch_model.safetensors")
    )


def _download_location(repo_id: str) -> tuple[str, str]:
    source = f"https://huggingface.co/{repo_id}"
    destination = str(Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}")
    return source, destination


def is_krea2_checkpoint(path: str) -> bool:
    """Identify the original transformer layout without relying on a filename."""
    if Path(path).suffix.lower() != ".safetensors":
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = set(checkpoint.keys())
    except (OSError, SafetensorError):
        return False
    return {"first.weight", "last.linear.weight", "txtfusion.projector.weight"} <= keys


def _diffusers_key(key: str) -> str:
    prefixes = {
        "first.": "img_in.",
        "tmlp.0.": "time_embed.linear_1.",
        "tmlp.2.": "time_embed.linear_2.",
        "tproj.1.": "time_mod_proj.",
        "txtmlp.0.scale": "txt_in.norm.weight",
        "txtmlp.1.": "txt_in.linear_1.",
        "txtmlp.3.": "txt_in.linear_2.",
        "txtfusion.": "text_fusion.",
        "blocks.": "transformer_blocks.",
        "last.linear.": "final_layer.linear.",
        "last.norm.scale": "final_layer.norm.weight",
        "last.modulation.lin": "final_layer.scale_shift_table",
    }
    for source, target in prefixes.items():
        if key.startswith(source):
            key = target + key[len(source) :]
            break

    key = key.replace(".prenorm.scale", ".norm1.weight")
    key = key.replace(".postnorm.scale", ".norm2.weight")
    key = key.replace(".mod.lin", ".scale_shift_table")
    key = key.replace(".mlp.", ".ff.")
    key = key.replace(".attn.wq.", ".attn.to_q.")
    key = key.replace(".attn.wk.", ".attn.to_k.")
    key = key.replace(".attn.wv.", ".attn.to_v.")
    key = key.replace(".attn.wo.", ".attn.to_out.0.")
    key = key.replace(".attn.gate.", ".attn.to_gate.")
    key = key.replace(".attn.qknorm.qnorm.scale", ".attn.norm_q.weight")
    key = key.replace(".attn.qknorm.knorm.scale", ".attn.norm_k.weight")
    return key


def _checkpoint_keys(checkpoint: safe_open, expected: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Match model weights and their optional ComfyUI scaled-FP8 companions."""
    keys = set(checkpoint.keys())
    scale_sources = {key.removesuffix("_scale"): key for key in keys if key.endswith(".weight_scale")}
    weight_sources = keys - set(scale_sources.values())
    mapped = {_diffusers_key(key): key for key in weight_sources}
    if set(mapped) != expected or len(mapped) != len(weight_sources):
        missing = sorted(expected - set(mapped))
        unexpected = sorted(set(mapped) - expected)
        raise ValueError(
            f"Unsupported Krea 2 checkpoint layout: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    metadata = checkpoint.metadata() or {}
    quantization = json.loads(metadata.get("_quantization_metadata", "{}"))
    layers = quantization.get("layers", {})
    scaled_layers = {source.removesuffix(".weight") for source in scale_sources}
    if set(layers) != scaled_layers or any(
        config.get("format") != "float8_e4m3fn" for config in layers.values()
    ):
        raise ValueError("Unsupported Krea 2 quantization metadata; expected scaled float8_e4m3fn weights.")
    if not set(scale_sources) <= weight_sources:
        raise ValueError("Krea 2 checkpoint has a weight_scale without a matching weight.")
    return mapped, scale_sources


def load_krea2_transformer(path: str) -> Krea2Transformer2DModel:
    """Stream each tensor into a meta-initialized model to avoid a second 26 GB copy."""
    with init_empty_weights():
        transformer = Krea2Transformer2DModel()

    expected_shapes = {key: value.shape for key, value in transformer.state_dict().items()}
    expected = set(expected_shapes)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        mapped, scale_sources = _checkpoint_keys(checkpoint, expected)

        for target, source in mapped.items():
            value = checkpoint.get_tensor(source)
            shape = expected_shapes[target]
            if value.shape != shape:
                if target.endswith(".scale_shift_table") and value.numel() == shape.numel():
                    value = value.reshape(shape)
                else:
                    raise ValueError(f"Invalid Krea 2 tensor shape for {source}: {tuple(value.shape)}")
            if source in scale_sources:
                scale = checkpoint.get_tensor(scale_sources[source])
                if value.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or scale.ndim != 0:
                    raise ValueError(f"Invalid Krea 2 scaled FP8 weight: {source}")
                value = value.to(torch.float32).mul_(scale).to(torch.bfloat16)
            elif value.dtype == torch.float8_e4m3fn:
                raise ValueError(f"Krea 2 FP8 weight is missing its scale: {source}")
            if target.endswith((".norm1.weight", ".norm2.weight", ".norm.weight", ".norm_q.weight", ".norm_k.weight")):
                dtype = torch.float32
            else:
                dtype = torch.bfloat16
            set_module_tensor_to_device(transformer, target, "cpu", value=value, dtype=dtype)

    return transformer.eval().requires_grad_(False)


def load_krea2_pipeline(
    path: str,
    device: torch.device | str,
    on_stage: Callable[[str, str | None, str | None], None] | None = None,
) -> Krea2Pipeline:
    """Use the selected local transformer and download only its shared components."""
    def stage(message: str, download_repo: str | None = None) -> None:
        if on_stage is not None:
            source, destination = _download_location(download_repo) if download_repo else (None, None)
            on_stage(message, source, destination)

    text_download = _text_encoder_needs_download()
    stage("Qwen3-VL 4B をダウンロード・読み込み中" if text_download else "Qwen3-VL 4B を読み込み中",
          TEXT_ENCODER_ID if text_download else None)
    text_encoder = Qwen3VLModel.from_pretrained(TEXT_ENCODER_ID, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID)
    vae_download = _vae_needs_download()
    stage("Qwen-Image VAE をダウンロード・読み込み中" if vae_download else "Qwen-Image VAE を読み込み中",
          VAE_ID if vae_download else None)
    vae = AutoencoderKLQwenImage.from_pretrained(VAE_ID, subfolder="vae", torch_dtype=torch.bfloat16)
    stage("Krea 2 の重みを読み込み中")
    transformer = load_krea2_transformer(path)
    stage("パイプラインを準備中")
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    pipeline = Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        is_distilled=True,
    )
    if str(device) == "cuda":
        stage("GPU オフロードを設定中")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    return pipeline
