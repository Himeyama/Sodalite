"""Lists generated images found in the output directory, with embedded generation metadata."""

from pathlib import Path

from PIL import UnidentifiedImageError
from pydantic import ValidationError

from sodalite_backend.imaging.png_metadata import read_metadata
from sodalite_backend.imaging.storage import output_dir as default_output_dir
from sodalite_backend.schemas.generation import GalleryImageInfo, GalleryParameters

IMAGE_EXTENSIONS = {".png"}


def list_gallery_images(output_dir: Path | None = None) -> list[GalleryImageInfo]:
    """List generated images directly under `output_dir` (non-recursive), newest first.

    Files that fail to open are skipped (most likely a truncated write from an
    interrupted generation). Files that open fine but carry no Sodalite
    metadata are still listed, with `parameters` left `None`.
    """
    directory = output_dir if output_dir is not None else default_output_dir()
    if not directory.is_dir():
        return []

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images = [info for path in files if (info := _build_info(path)) is not None]
    return sorted(images, key=lambda image: image.created_at, reverse=True)


def _build_info(path: Path) -> GalleryImageInfo | None:
    try:
        raw_parameters = read_metadata(path)
    except (UnidentifiedImageError, OSError):
        return None

    parameters = None
    if raw_parameters is not None:
        try:
            parameters = GalleryParameters.model_validate(raw_parameters)
        except ValidationError:
            parameters = None

    stat = path.stat()
    return GalleryImageInfo(
        image_id=path.name,
        image_url=f"/api/v1/images/{path.name}",
        image_path=str(path.resolve()),
        created_at=stat.st_mtime,
        parameters=parameters,
    )
