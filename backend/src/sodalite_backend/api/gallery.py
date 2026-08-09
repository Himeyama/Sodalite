"""Generation history gallery: lists and deletes images in the output directory."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from sodalite_backend.imaging.gallery import list_gallery_images
from sodalite_backend.imaging.storage import output_dir
from sodalite_backend.schemas.generation import GalleryImageInfo

router = APIRouter(prefix="/gallery", tags=["gallery"])


@router.get("/images", response_model=list[GalleryImageInfo])
def list_images() -> list[GalleryImageInfo]:
    return list_gallery_images()


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: str) -> None:
    path = _resolve_image_path(image_id)
    try:
        path.unlink()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Image not found.") from error


def _resolve_image_path(image_id: str) -> Path:
    """Resolve `image_id` (a bare filename) to a path inside the output directory,
    rejecting anything that would escape the directory (path traversal)."""
    resolved_dir = output_dir().resolve()
    path = (resolved_dir / image_id).resolve()
    if path.parent != resolved_dir or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Image not found.")
    return path
