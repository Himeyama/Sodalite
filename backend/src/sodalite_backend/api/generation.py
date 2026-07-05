"""Text-to-image generation endpoint.

Phase 2 runs generation synchronously within the request, but the response
shape already carries a `job_id` so the API can move to async job polling
(Phase 3) without a breaking change.
"""

import uuid

from fastapi import APIRouter, Request

from sodalite_backend.imaging.png_metadata import save_with_metadata
from sodalite_backend.imaging.storage import new_image_path
from sodalite_backend.schemas.generation import GenerationJob, TextToImageRequest

router = APIRouter(prefix="/generations", tags=["generations"])


@router.post("/text-to-image", response_model=GenerationJob)
def create_text_to_image(request: Request, body: TextToImageRequest) -> GenerationJob:
    pipeline_manager = request.app.state.pipeline_manager

    image = pipeline_manager.generate(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        steps=body.steps,
        cfg_scale=body.cfg_scale,
        width=body.width,
        height=body.height,
        sampler=body.sampler,
        seed=body.seed,
        loras=body.loras,
    )

    image_path = new_image_path()
    save_with_metadata(image, image_path, body.model_dump())

    return GenerationJob(
        job_id=uuid.uuid4().hex,
        status="completed",
        progress=1.0,
        current_step=body.steps,
        total_steps=body.steps,
        image_url=f"/api/v1/images/{image_path.name}",
    )
