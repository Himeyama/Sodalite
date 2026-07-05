"""Lists the text-to-image models the app offers: known HF repos and imported single files.

The app deliberately does *not* enumerate the entire Hugging Face cache — that surfaces
unrelated repos the user never chose. It lists only the repos it tracks as base models
(see [known_hf_models_store]) plus locally imported checkpoints, always flagging the
active one.
"""

from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import CachedRepoInfo

from sodalite_backend.inference.imported_models_store import load_imported_model_paths
from sodalite_backend.inference.known_hf_models_store import load_known_hf_model_ids
from sodalite_backend.schemas.generation import ModelInfo


def list_cached_models(active_model_id: str) -> list[ModelInfo]:
    """List the offered models (known HF repos + imported single files), flagging the active one."""
    models = _list_hf_models(active_model_id) + _list_imported_models(active_model_id)
    if not any(model.is_active for model in models):
        models.append(ModelInfo(model_id=active_model_id, is_active=True, size_on_disk_bytes=0))
    return sorted(models, key=lambda model: model.model_id)


def _list_hf_models(active_model_id: str) -> list[ModelInfo]:
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
