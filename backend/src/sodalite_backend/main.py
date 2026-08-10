"""FastAPI application entry point for the Sodalite backend."""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sodalite_backend.api.router import api_router
from sodalite_backend.config import load_config
from sodalite_backend.imaging.storage import output_dir
from sodalite_backend.inference.active_model_store import load_active_model_id
from sodalite_backend.inference.job_manager import JobManager
from sodalite_backend.inference.known_hf_models_store import add_known_hf_model_id
from sodalite_backend.inference.pipeline_manager import PipelineManager

WEBUI_DIR = Path(__file__).parent / "webui"

# Last-resort fallback used only when there is neither a saved model
# (active_model_store) nor an explicit --model-id: a truly first run.
DEFAULT_MODEL_ID = "stabilityai/sd-turbo"


def _model_id_still_available(model_id: str | None) -> bool:
    """A saved HF repo id is always worth trying; a saved local checkpoint must still exist."""
    if model_id is None:
        return False
    path = Path(model_id)
    return not path.is_absolute() or path.is_file()


def create_app(model_id: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The user's last-selected model (if any) takes priority over an
        # explicit --model-id so a restart resumes where they left off. A saved
        # local checkpoint path that no longer exists is ignored rather than
        # left to blow up pipeline loading.
        saved_model_id = load_active_model_id()
        startup_model_id = saved_model_id if _model_id_still_available(saved_model_id) else model_id
        app.state.pipeline_manager = PipelineManager(startup_model_id)
        app.state.job_manager = JobManager(app.state.pipeline_manager)
        # A local file is an imported checkpoint; a repo id is a known HF model to
        # keep in the list. Either way the startup model is always offered.
        if not Path(startup_model_id).is_file():
            add_known_hf_model_id(startup_model_id)
        yield

    app = FastAPI(title="Sodalite Backend", lifespan=lifespan)
    app.include_router(api_router)

    app.mount("/api/v1/images", StaticFiles(directory=output_dir()), name="images")

    # Serve the browser webui from the app root. Mounting "/" swallows every
    # unmatched path, so this must come last, after the API router and the
    # images mount above. `html=True` serves index.html for "/".
    app.mount("/", StaticFiles(directory=WEBUI_DIR, html=True), name="webui")

    return app


def run() -> None:
    config = load_config()
    app = create_app(config.model_id or DEFAULT_MODEL_ID)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    run()
