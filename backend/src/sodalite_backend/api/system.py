"""Health and system-info endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from sodalite_backend.inference.active_model_store import save_active_model_id
from sodalite_backend.inference.directories_store import (
    ScanDirectories,
    load_directories,
    save_directories,
)
from sodalite_backend.inference.known_hf_models_store import add_known_hf_model_id
from sodalite_backend.inference.lora_registry import list_imported_loras
from sodalite_backend.inference.model_registry import list_cached_models
from sodalite_backend.inference.samplers import available_samplers
from sodalite_backend.schemas.generation import (
    LoraFileInfo,
    ModelInfo,
    ScanDirectoriesInfo,
    SetActiveModelRequest,
)

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    pipeline_manager = request.app.state.pipeline_manager
    return {
        "status": "ok",
        "device": pipeline_manager.device_backend,
        "loaded_model": pipeline_manager.model_id,
        "model_ready": pipeline_manager.is_ready,
    }


@router.get("/samplers")
def samplers() -> list[str]:
    return available_samplers()


@router.get("/models")
def models(request: Request) -> list[ModelInfo]:
    pipeline_manager = request.app.state.pipeline_manager
    return list_cached_models(pipeline_manager.model_id)


@router.post("/models/active")
def set_active_model(request: Request, body: SetActiveModelRequest) -> ModelInfo:
    pipeline_manager = request.app.state.pipeline_manager
    if not pipeline_manager.is_ready:
        raise HTTPException(status_code=503, detail="The initial model is still loading.")
    try:
        pipeline_manager.load_model(body.model_id)
    except OSError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # Remember HF repos the user activates so they keep showing in the list;
    # single-file checkpoints are surfaced by scanning the configured model directory.
    if not Path(body.model_id).is_file():
        add_known_hf_model_id(body.model_id)

    # Persist so the next backend start resumes with this model instead of the default.
    save_active_model_id(pipeline_manager.model_id)

    return ModelInfo(model_id=pipeline_manager.model_id, is_active=True, size_on_disk_bytes=0)


@router.get("/loras")
def loras() -> list[LoraFileInfo]:
    return list_imported_loras()


@router.get("/settings/directories")
def get_directories() -> ScanDirectoriesInfo:
    directories = load_directories()
    return ScanDirectoriesInfo(model_dir=directories.model_dir, lora_dir=directories.lora_dir)


@router.put("/settings/directories")
def put_directories(body: ScanDirectoriesInfo) -> ScanDirectoriesInfo:
    """Set the directories scanned for checkpoints and LoRA files.

    A null value leaves that directory unset. A non-null value must point to an
    existing directory.
    """
    _validate_directory(body.model_dir)
    _validate_directory(body.lora_dir)

    saved = save_directories(ScanDirectories(model_dir=body.model_dir, lora_dir=body.lora_dir))
    return ScanDirectoriesInfo(model_dir=saved.model_dir, lora_dir=saved.lora_dir)


def _validate_directory(directory: str | None) -> None:
    if directory is not None and not Path(directory).is_dir():
        raise HTTPException(status_code=422, detail=f"Directory not found: {directory}")
