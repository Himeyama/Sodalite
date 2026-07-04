"""Discovers text-to-image models available locally: Hugging Face cache and imported single files."""

from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CachedRepoInfo

from sdapp_backend.inference.imported_models_store import load_imported_model_paths
from sdapp_backend.schemas.generation import ModelInfo


def list_cached_models(active_model_id: str) -> list[ModelInfo]:
    """List locally available models (HF cache + imported single files), flagging the active one."""
    models = _list_hf_cached_models(active_model_id) + _list_imported_models(active_model_id)
    if not any(model.is_active for model in models):
        models.append(ModelInfo(model_id=active_model_id, is_active=True, size_on_disk_bytes=0))
    return sorted(models, key=lambda model: model.model_id)


def _list_hf_cached_models(active_model_id: str) -> list[ModelInfo]:
    cache_info = scan_cache_dir()
    return [
        ModelInfo(
            model_id=repo.repo_id,
            is_active=repo.repo_id == active_model_id,
            size_on_disk_bytes=repo.size_on_disk,
        )
        for repo in cache_info.repos
        if repo.repo_type == "model" and _has_pipeline_files(repo)
    ]


def _list_imported_models(active_model_id: str) -> list[ModelInfo]:
    return [
        ModelInfo(
            model_id=path,
            is_active=path == active_model_id,
            size_on_disk_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
        )
        for path in load_imported_model_paths()
    ]


def _has_pipeline_files(repo: CachedRepoInfo) -> bool:
    file_names = {file.file_name for revision in repo.revisions for file in revision.files}
    return "model_index.json" in file_names
