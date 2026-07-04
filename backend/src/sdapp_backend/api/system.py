"""Health and system-info endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from sdapp_backend.inference.imported_models_store import add_imported_model_path
from sdapp_backend.inference.model_registry import list_cached_models
from sdapp_backend.inference.samplers import available_samplers
from sdapp_backend.schemas.generation import ImportModelRequest, ModelInfo, SetActiveModelRequest

SUPPORTED_CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt"}

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    pipeline_manager = request.app.state.pipeline_manager
    return {
        "status": "ok",
        "device": pipeline_manager.device,
        "loaded_model": pipeline_manager.model_id,
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
    try:
        pipeline_manager.load_model(body.model_id)
    except OSError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ModelInfo(model_id=pipeline_manager.model_id, is_active=True, size_on_disk_bytes=0)


@router.post("/models/imported")
def import_model(body: ImportModelRequest) -> ModelInfo:
    path = Path(body.model_path)
    if path.suffix.lower() not in SUPPORTED_CHECKPOINT_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {path.suffix}")
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f"File not found: {body.model_path}")

    add_imported_model_path(body.model_path)
    return ModelInfo(model_id=body.model_path, is_active=False, size_on_disk_bytes=path.stat().st_size)
