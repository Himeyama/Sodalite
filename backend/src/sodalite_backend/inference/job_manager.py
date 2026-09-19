"""Runs text-to-image batches on a background thread and tracks their progress.

Generation blocks the GPU for anywhere from seconds to minutes, so a batch runs
on its own thread rather than the request handler. The frontend polls
`GET /generations/{job_id}` for the latest snapshot instead of waiting on the
original POST, which lets it show each image as soon as it's saved and offer
a cancel button that takes effect between images.
"""

from base64 import b64decode
from io import BytesIO
import threading
import uuid
from typing import Protocol

from PIL import Image

from sodalite_backend.imaging.png_metadata import save_with_metadata
from sodalite_backend.imaging.storage import new_image_path
from sodalite_backend.schemas.generation import (
    GenerationJob,
    ImageToImageRequest,
    LoraSpec,
    Sampler,
    TextToImageRequest,
)


class SupportsGenerate(Protocol):
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
        sampler: Sampler,
        seed: int | None,
        batch_size: int,
        loras: list[LoraSpec] | None,
        initial_image: Image.Image | None,
        strength: float,
        should_stop,
        on_step,
    ): ...


class JobManager:
    """Owns the in-memory table of generation jobs and the threads running them.

    Jobs are kept in memory only (no persistence): if the backend restarts,
    in-flight jobs are simply gone, which matches today's single-user,
    single-session usage. A `threading.Lock` guards the shared dict since the
    worker thread and the polling request handler both touch it.
    """

    def __init__(self, pipeline_manager: SupportsGenerate) -> None:
        self._pipeline_manager = pipeline_manager
        self._jobs: dict[str, GenerationJob] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start_text_to_image(self, request: TextToImageRequest) -> GenerationJob:
        return self._start_job(request)

    def start_image_to_image(self, request: ImageToImageRequest) -> GenerationJob:
        return self._start_job(request)

    def _start_job(self, request: TextToImageRequest | ImageToImageRequest) -> GenerationJob:
        job_id = uuid.uuid4().hex

        job = GenerationJob(
            job_id=job_id,
            status="queued",
            total_steps=request.steps,
            total_images=request.batch_size,
        )
        cancel_event = threading.Event()

        with self._lock:
            self._jobs[job_id] = job
            self._cancel_flags[job_id] = cancel_event

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, request, cancel_event),
            daemon=True,
        )
        thread.start()

        return job

    def get_job(self, job_id: str) -> GenerationJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> GenerationJob | None:
        with self._lock:
            cancel_event = self._cancel_flags.get(job_id)
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if cancel_event is not None:
                cancel_event.set()
            return job

    def _run_job(
        self,
        job_id: str,
        request: TextToImageRequest | ImageToImageRequest,
        cancel_event: threading.Event,
    ) -> None:
        self._update_job(job_id, status="running")

        def report_step(step_index: int, total_steps: int) -> None:
            self._update_job(job_id, current_step=step_index, total_steps=total_steps)

        try:
            initial_image = (
                _decode_initial_image(request.initial_image).resize(
                    (request.width, request.height), Image.Resampling.LANCZOS
                )
                if isinstance(request, ImageToImageRequest)
                else None
            )
            images = self._pipeline_manager.generate(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                width=request.width,
                height=request.height,
                sampler=request.sampler,
                seed=request.seed,
                batch_size=request.batch_size,
                loras=request.loras,
                initial_image=initial_image,
                strength=request.strength if isinstance(request, ImageToImageRequest) else 0.4,
                should_stop=cancel_event.is_set,
                on_step=report_step,
            )

            # Do not embed the (potentially multi-megabyte) source image in every output PNG.
            metadata = request.model_dump(exclude={"initial_image"})
            images_completed = 0
            for image in images:
                image_path = new_image_path()
                save_with_metadata(image, image_path, metadata)
                images_completed += 1

                self._update_job(
                    job_id,
                    status="running",
                    progress=images_completed / request.batch_size,
                    current_step=request.steps,
                    total_steps=request.steps,
                    images_completed=images_completed,
                    image_url=f"/api/v1/images/{image_path.name}",
                    image_path=str(image_path.resolve()),
                )

            final_status = "cancelled" if cancel_event.is_set() else "completed"
            self._update_job(
                job_id, status=final_status, progress=images_completed / request.batch_size
            )
        except Exception as exc:  # Any failure (OOM, bad LoRA, etc.) surfaces via the job status.
            self._update_job(job_id, status="failed", error=str(exc))

    def _update_job(self, job_id: str, **changes: object) -> None:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return
            self._jobs[job_id] = current.model_copy(update=changes)


def _decode_initial_image(encoded_image: str) -> Image.Image:
    """Decode a client-provided base64 image and detach it from the input stream."""
    try:
        image_bytes = b64decode(encoded_image, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB").copy()
    except Exception as exc:
        raise ValueError("The selected image is not a valid image file.") from exc
