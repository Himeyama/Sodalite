"""Lists generated images found in the output directory, with embedded generation metadata."""

from pathlib import Path

from PIL import UnidentifiedImageError
from pydantic import ValidationError

from sodalite_backend.imaging.png_metadata import read_metadata
from sodalite_backend.imaging.storage import output_dir as default_output_dir
from sodalite_backend.schemas.generation import GalleryImageInfo, GalleryParameters

IMAGE_EXTENSIONS = {".png"}

# Keyed by resolved path -> ((mtime, size) used to build the entry, the entry itself).
# Reading PNG tEXt metadata means opening every file, which gets slow as the
# gallery grows; most calls re-list a directory that barely changed, so we
# skip re-reading any file whose (mtime, size) we've already seen. Size is
# included alongside mtime since mtime alone has only ~1s resolution on some
# filesystems and could miss a same-second delete-and-recreate.
_cache: dict[Path, tuple[tuple[float, int], GalleryImageInfo]] = {}


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

    seen = {path.resolve() for path in files}
    for stale in _cache.keys() - seen:
        del _cache[stale]

    images = [info for path in files if (info := _cached_info(path)) is not None]
    return sorted(images, key=lambda image: image.created_at, reverse=True)


def _cached_info(path: Path) -> GalleryImageInfo | None:
    resolved = path.resolve()
    stat = path.stat()
    fingerprint = (stat.st_mtime, stat.st_size)

    if (cached := _cache.get(resolved)) is not None and cached[0] == fingerprint:
        return cached[1]

    info = _build_info(path, stat.st_mtime)
    if info is not None:
        _cache[resolved] = (fingerprint, info)
    else:
        _cache.pop(resolved, None)
    return info


def _build_info(path: Path, mtime: float) -> GalleryImageInfo | None:
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

    return GalleryImageInfo(
        image_id=path.name,
        image_url=f"/api/v1/images/{path.name}",
        image_path=str(path.resolve()),
        created_at=mtime,
        parameters=parameters,
    )
