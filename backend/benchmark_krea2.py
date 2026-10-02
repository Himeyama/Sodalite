"""Measure Krea 2 on a real CUDA/ROCm GPU and retain images and JSON evidence.

Run with the existing ROCm environment, for example:
    uv run --no-sync python benchmark_krea2.py --model <checkpoint> --output outputs/krea2-optimized
Use --baseline after optimization to restore Diffusers' original model offload.
"""

import argparse
import faulthandler
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sodalite_backend.inference.krea2 import load_krea2_pipeline
from sodalite_backend.inference.pipeline_manager import PipelineManager


def main() -> None:
    """Generate identical seeded images and report latency, memory and GPU execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--expand-fp8", action="store_true", help="Expand FP8 on load for an exact-weight reference"
    )
    parser.add_argument(
        "--trace-after", type=int, default=0, help="Dump stacks while diagnosing a slow stage"
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="A red fox sitting in a meadow of wildflowers, morning sunlight, detailed photograph",
    )
    args = parser.parse_args()
    # Match Sodalite's ROCm launchers; unrestricted MIOpen tuning can take
    # minutes and would otherwise dominate a first-run comparison.
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("MIOPEN_FIND_ENFORCE", "NONE")
    if args.trace_after:
        faulthandler.dump_traceback_later(args.trace_after, repeat=True)
    if not torch.cuda.is_available():
        parser.error(
            "A real CUDA/ROCm GPU is required; CPU inference is not a benchmark substitute."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipeline = load_krea2_pipeline(
        args.model,
        "cuda",
        lambda stage, *_: print(stage, flush=True),
        optimize=not args.baseline,
        keep_scaled_fp8=not args.expand_fp8,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"Loaded in {load_seconds:.3f}s", flush=True)
    manager = PipelineManager()
    manager._pipeline = pipeline
    manager.model_id = args.model
    execution_devices: dict[str, list[str]] = {
        "text_encoder": [],
        "transformer": [],
        "vae_decoder": [],
    }
    handles = []
    for name in execution_devices:

        def record_device(module, _inputs, _output, name=name):
            # Group offload may already have returned weights to CPU by this
            # point. Record the actual result tensor, not its storage location.
            result = _output
            if isinstance(result, tuple):
                result = result[0]
            elif hasattr(result, "last_hidden_state"):
                result = result.last_hidden_state
            elif hasattr(result, "sample"):
                result = result.sample
            execution_devices[name].append(str(result.device))

        module = pipeline.vae.decoder if name == "vae_decoder" else getattr(pipeline, name)
        handles.append(module.register_forward_hook(record_device))
        if name == "vae_decoder":
            handles.append(
                module.register_forward_pre_hook(
                    lambda *_: print("VAE GPU decoding started", flush=True)
                )
            )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    image_seconds = []
    previous = started

    def report_step(step: int, total: int) -> None:
        torch.cuda.synchronize()
        print(
            f"GPU step {step}/{total} finished at {time.perf_counter() - started:.3f}s", flush=True
        )

    try:
        for index, image in enumerate(
            manager.generate(
                prompt=args.prompt,
                negative_prompt="",
                steps=args.steps,
                cfg_scale=0,
                width=args.size,
                height=args.size,
                sampler="euler",
                seed=args.seed,
                batch_size=args.images,
                on_step=report_step,
            )
        ):
            torch.cuda.synchronize()
            now = time.perf_counter()
            image_seconds.append(now - previous)
            previous = now
            image.save(args.output / f"seed-{args.seed + index}.png")
            pixels = np.asarray(image)
            if not pixels.std() > 1:
                raise RuntimeError("Generated an empty/constant image.")
            print(
                f"Image {index + 1}: {image_seconds[-1]:.3f}s, std={pixels.std():.3f}", flush=True
            )
    finally:
        for handle in handles:
            handle.remove()
    result = {
        "model": args.model,
        "baseline": args.baseline,
        "expand_fp8": args.expand_fp8,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "size": args.size,
        "load_seconds": load_seconds,
        "image_seconds": image_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
        "execution_devices": execution_devices,
    }
    (args.output / "measurements.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
