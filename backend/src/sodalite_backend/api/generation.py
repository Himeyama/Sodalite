"""Text-to-image generation endpoints.

Generation runs on a background thread owned by `JobManager`: the POST below
starts the job and returns immediately with a `job_id`, and the frontend
polls `GET /{job_id}` for progress and each completed image as the batch
runs. `DELETE /{job_id}` requests cancellation, which takes effect before the
next image in the batch starts.
"""

from fastapi import APIRouter, HTTPException, Request

from sodalite_backend.schemas.generation import (
    GenerationJob,
    ImageToImageRequest,
    TextToImageRequest,
)

router = APIRouter(prefix="/generations", tags=["generations"])


@router.post("/text-to-image", response_model=GenerationJob)
def create_text_to_image(request: Request, body: TextToImageRequest) -> GenerationJob:
    job_manager = request.app.state.job_manager
    return job_manager.start_text_to_image(body)


@router.post("/image-to-image", response_model=GenerationJob)
def create_image_to_image(request: Request, body: ImageToImageRequest) -> GenerationJob:
    """Start a generation that uses the supplied image as its initial latent."""
    job_manager = request.app.state.job_manager
    return job_manager.start_image_to_image(body)


@router.get("/{job_id}", response_model=GenerationJob)
def get_generation_job(request: Request, job_id: str) -> GenerationJob:
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}", response_model=GenerationJob)
def cancel_generation_job(request: Request, job_id: str) -> GenerationJob:
    job_manager = request.app.state.job_manager
    job = job_manager.cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
