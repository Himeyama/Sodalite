"""FastAPI application entry point for the Sodalite backend."""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sodalite_backend.api.router import api_router
from sodalite_backend.config import load_config
from sodalite_backend.imaging.storage import OUTPUT_DIR
from sodalite_backend.inference.known_hf_models_store import add_known_hf_model_id
from sodalite_backend.inference.pipeline_manager import PipelineManager


def create_app(model_id: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline_manager = PipelineManager(model_id)
        # A local file is an imported checkpoint; a repo id is a known HF model to
        # keep in the list. Either way the startup model is always offered.
        if not Path(model_id).is_file():
            add_known_hf_model_id(model_id)
        yield

    app = FastAPI(title="Sodalite Backend", lifespan=lifespan)
    app.include_router(api_router)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/api/v1/images", StaticFiles(directory=OUTPUT_DIR), name="images")

    return app


def run() -> None:
    config = load_config()
    app = create_app(config.model_id)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    run()
