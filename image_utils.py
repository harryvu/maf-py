from __future__ import annotations

import base64
import mimetypes
import pathlib
from urllib.parse import urlparse

from agent_framework import UriContent


def uri_content_from_image_source(source: str) -> UriContent:
    """Create a UriContent from an image source.

    Supported sources:
    - Local file path (converted to a data: URI)
    - http(s) URL
    - data: URI (passed through)
    """

    cleaned = source.strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("Image source is empty")

    lowered = cleaned.lower()

    if lowered.startswith("data:"):
        # Best-effort media type extraction.
        media_type = "application/octet-stream"
        try:
            header = cleaned[5 : cleaned.index(",")]
            if ";" in header:
                media_type = header.split(";", 1)[0] or media_type
            else:
                media_type = header or media_type
        except ValueError:
            pass
        return UriContent(uri=cleaned, media_type=media_type)

    if lowered.startswith("http://") or lowered.startswith("https://"):
        guessed = mimetypes.guess_type(urlparse(cleaned).path)[0]
        media_type = guessed or "image/jpeg"
        if not media_type.startswith("image/"):
            # Some URLs don't include an extension; default to a common image type.
            media_type = "image/jpeg"
        return UriContent(uri=cleaned, media_type=media_type)

    path = pathlib.Path(cleaned)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {cleaned}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {cleaned}")

    guessed = mimetypes.guess_type(path.name)[0]
    media_type = guessed or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise ValueError(
            f"Unsupported media type '{media_type}'. Please provide an image file (png/jpg/webp/etc)."
        )

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    data_uri = f"data:{media_type};base64,{encoded}"
    return UriContent(uri=data_uri, media_type=media_type)
