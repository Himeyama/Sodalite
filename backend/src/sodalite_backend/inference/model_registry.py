"""Lists the text-to-image models the app offers: known HF repos and scanned checkpoints.

The app deliberately does *not* enumerate the entire Hugging Face cache — that surfaces
unrelated repos the user never chose. It lists only the repos it tracks as base models
(see [known_hf_models_store]) plus any single-file checkpoints found by scanning the
user-configured model directory, always flagging the active one.
"""

from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CachedRepoInfo

from sodalite_backend.inference.directories_store import load_directories
from sodalite_backend.inference.known_hf_models_store import load_known_hf_model_ids
from sodalite_backend.schemas.generation import ModelInfo

CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt"}


def list_cached_models(active_model_id: str | None) -> list[ModelInfo]:
    """List the offered models (known HF repos + scanned checkpoints), flagging the active one.

    `active_model_id` is `None` while the initial model is still loading in the
    background; in that case nothing is flagged active.
    """
    models = _list_hf_models(active_model_id) + _list_directory_models(active_model_id)
    if active_model_id is not None and not any(model.is_active for model in models):
        models.append(ModelInfo(model_id=active_model_id, is_active=True, size_on_disk_bytes=0))
    return sorted(models, key=lambda model: model.model_id)


def _list_hf_models(active_model_id: str | None) -> list[ModelInfo]:
    """Known HF repos that are actually present in the cache as a usable pipeline."""
    known_ids = set(load_known_hf_model_ids())
    sizes = {
        repo.repo_id: repo.size_on_disk
        for repo in scan_cache_dir().repos
        if repo.repo_type == "model" and repo.repo_id in known_ids and _has_pipeline_files(repo)
    }
    return [
        ModelInfo(
            model_id=repo_id,
            is_active=repo_id == active_model_id,
            size_on_disk_bytes=size,
        )
        for repo_id, size in sizes.items()
    ]


def _list_directory_models(active_model_id: str | None) -> list[ModelInfo]:
    """Single-file checkpoints found by scanning the configured model directory."""
    model_dir = load_directories().model_dir
    if model_dir is None:
        return []

    return [
        ModelInfo(
            model_id=str(path),
            is_active=str(path) == active_model_id,
            size_on_disk_bytes=path.stat().st_size,
        )
        for path in scan_checkpoint_files(Path(model_dir))
    ]


def scan_checkpoint_files(directory: Path) -> list[Path]:
    """Find supported checkpoint files directly under directory (non-recursive), sorted by path."""
    if not directory.is_dir():
        return []

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]
    return sorted(files)


def _has_pipeline_files(repo: CachedRepoInfo) -> bool:
    file_names = {file.file_name for revision in repo.revisions for file in revision.files}
    return "model_index.json" in file_names
