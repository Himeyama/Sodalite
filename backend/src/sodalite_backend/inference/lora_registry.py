"""Lists locally imported LoRA files."""

from pathlib import Path

from sodalite_backend.inference.imported_loras_store import load_imported_lora_paths
from sodalite_backend.schemas.generation import LoraFileInfo


def list_imported_loras() -> list[LoraFileInfo]:
    """List imported LoRA files, sorted by path."""
    loras = [
        LoraFileInfo(
            lora_id=path,
            size_on_disk_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
        )
        for path in load_imported_lora_paths()
    ]
    return sorted(loras, key=lambda lora: lora.lora_id)
