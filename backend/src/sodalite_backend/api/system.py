"""Health and system-info endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from sodalite_backend.inference.imported_loras_store import add_imported_lora_path
from sodalite_backend.inference.imported_models_store import (
    add_imported_model_path,
    load_imported_model_paths,
    remove_imported_model_path,
)
from sodalite_backend.inference.known_hf_models_store import add_known_hf_model_id
from sodalite_backend.inference.lora_registry import list_imported_loras
from sodalite_backend.inference.model_registry import list_cached_models
from sodalite_backend.inference.samplers import available_samplers
from sodalite_backend.schemas.generation import (
    ImportLoraRequest,
    ImportModelRequest,
    LoraFileInfo,
    ModelInfo,
    SetActiveModelRequest,
)

SUPPORTED_CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt"}
SUPPORTED_LORA_EXTENSIONS = {".safetensors"}

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

    # Remember HF repos the user activates so they keep showing in the list;
    # imported single files are already tracked by imported_models_store.
    if not Path(body.model_id).is_file():
        add_known_hf_model_id(body.model_id)

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


@router.delete("/models/imported")
def remove_imported_model(request: Request, body: ImportModelRequest) -> dict[str, str]:
    """Drop an imported checkpoint from the registry without deleting the file on disk."""
    pipeline_manager = request.app.state.pipeline_manager
    if body.model_path == pipeline_manager.model_id:
        raise HTTPException(status_code=409, detail="Cannot remove the active model.")
    if body.model_path not in load_imported_model_paths():
        raise HTTPException(status_code=404, detail=f"Model not imported: {body.model_path}")

    remove_imported_model_path(body.model_path)
    return {"model_id": body.model_path}


@router.get("/loras")
def loras() -> list[LoraFileInfo]:
    return list_imported_loras()


@router.post("/loras/imported")
def import_lora(body: ImportLoraRequest) -> LoraFileInfo:
    path = Path(body.lora_path)
    if path.suffix.lower() not in SUPPORTED_LORA_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {path.suffix}")
    if not path.is_file():
        raise HTTPException(status_code=422, detail=f"File not found: {body.lora_path}")

    add_imported_lora_path(body.lora_path)
    return LoraFileInfo(lora_id=body.lora_path, size_on_disk_bytes=path.stat().st_size)
