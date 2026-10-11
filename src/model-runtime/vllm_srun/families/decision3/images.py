"""Decision 3.0 image inputs: decoding and the Qwen2-VL image processor, as the released runtime reads them.

A request's images are base64 PNG, JPEG or WebP data URLs of at most 8,000,000
bytes and 16,000,000 pixels (the released server's strict loading); PIL
decodes each and converts it to RGB, nothing else. The processor is the
Transformers 5.17 ``Qwen2VLImageProcessor`` torchvision backend with the
released runtime's pixel budget: ``smart_resize`` to a multiple of
``patch_size * merge_size`` with 65,536 to 1,638,400 pixels, torchvision's
uint8 bicubic antialiased resize, rescale and normalize fused into one FP32
subtraction and division, then the patch layout with the frame repeated over
the temporal patch. Each ``merge_size**2`` patches become one input token.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

# Preprocessing runs on one thread: PyTorch's OpenMP backend keeps a thread team
# per calling thread, and a team per request thread outnumbers the cores, so every
# parallel region of the resize and normalization would pay a wake-up.
_PREPROCESS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-sr-images")

MIN_PIXELS = 65_536
MAX_PIXELS = 1_638_400
MAX_IMAGE_BYTES = 8_000_000
MAX_SOURCE_PIXELS = 16_000_000
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
DATA_URL_FORMATS = ("png", "jpeg", "jpg", "webp")
MAX_ASPECT_RATIO = 200


@dataclass(frozen=True)
class ProcessorSettings:
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    rescale_factor: float = 1 / 255
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProcessorSettings:
        return cls(
            patch_size=config["patch_size"],
            temporal_patch_size=config["temporal_patch_size"],
            merge_size=config["merge_size"],
            image_mean=tuple(config["image_mean"]),
            image_std=tuple(config["image_std"]),
            rescale_factor=config.get("rescale_factor", 1 / 255),
        )

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


@dataclass(frozen=True)
class ProcessedImage:
    """One image as the vision tower reads it: FP32 patch rows, its (t, h, w) patch grid, its input tokens."""

    pixel_values: torch.Tensor
    grid: tuple[int, int, int]
    tokens: int
    digest: str


def _payload(value: Any) -> bytes:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ValueError("images must be base64 PNG, JPEG or WebP data URLs")
    header, separator, encoded = value.partition(",")
    kind = header.strip().lower()
    if (
        not separator
        or not kind.startswith("data:image/")
        or not kind.endswith(";base64")
    ):
        raise ValueError("a data URL image is data:image/<format>;base64,<data>")
    if kind[len("data:image/") : -len(";base64")] not in DATA_URL_FORMATS:
        raise ValueError("images must be base64 PNG, JPEG or WebP data URLs")
    if len(encoded) > 4 * -(-MAX_IMAGE_BYTES // 3):
        raise ValueError(f"each image must be at most {MAX_IMAGE_BYTES:,} bytes")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 image data") from exc
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError(f"each image must be at most {MAX_IMAGE_BYTES:,} bytes")
    return payload


def decode(value: Any) -> tuple[Any, str]:
    """An image data URL as an RGB PIL image and the SHA-256 of its encoded bytes; ValueError when invalid."""
    from PIL import Image, UnidentifiedImageError

    payload = _payload(value)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in IMAGE_FORMATS:
                raise ValueError("images must be PNG, JPEG or WebP")
            if image.width * image.height > MAX_SOURCE_PIXELS:
                raise ValueError(
                    f"each image must have at most {MAX_SOURCE_PIXELS:,} pixels"
                )
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB"), hashlib.sha256(payload).hexdigest()
    except (
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"invalid image data ({type(exc).__name__})") from exc


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Transformers' ``smart_resize``: both sides a multiple of ``factor``, the area within the budget."""
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def input_tokens(settings: ProcessorSettings, width: int, height: int) -> int:
    """The input tokens of one image of this size; ValueError past the aspect-ratio limit."""
    h, w = smart_resize(
        height, width, settings.factor, settings.min_pixels, settings.max_pixels
    )
    return (
        (h // settings.patch_size)
        * (w // settings.patch_size)
        // settings.merge_size**2
    )


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """torchvision's ``resize_image`` for a CPU uint8 batch, bicubic with antialiasing."""
    shape = image.shape
    numel = image.numel()
    channels, old_height, old_width = shape[-3:]
    if (height, width) == (old_height, old_width):
        return image
    image = image.reshape(-1, channels, old_height, old_width)
    strides = image.stride()
    if (
        image.is_contiguous(memory_format=torch.channels_last)
        and image.shape[0] == 1
        and numel != strides[0]
    ):
        restrided = list(strides)
        restrided[0] = numel
        image = image.as_strided((1, channels, old_height, old_width), restrided)
    image = F.interpolate(
        image, size=[height, width], mode="bicubic", align_corners=False, antialias=True
    )
    return image.reshape((*shape[:-3], channels, height, width))


def preprocess(image: Any, digest: str, settings: ProcessorSettings) -> ProcessedImage:
    """One RGB PIL image through the processor (on the preprocessing thread): FP32 patch rows and its grid."""
    return _PREPROCESS.submit(_preprocess, image, digest, settings).result()


def _preprocess(image: Any, digest: str, settings: ProcessorSettings) -> ProcessedImage:
    import numpy as np

    pixels = torch.as_tensor(np.array(image, copy=True))
    pixels = pixels.view(image.size[1], image.size[0], 3).permute(2, 0, 1)
    stacked = torch.stack([pixels])
    _, _, old_height, old_width = stacked.shape
    height, width = smart_resize(
        old_height, old_width, settings.factor, settings.min_pixels, settings.max_pixels
    )
    resized = _resize(stacked, height, width)
    mean = torch.tensor(settings.image_mean) * (1.0 / settings.rescale_factor)
    std = torch.tensor(settings.image_std) * (1.0 / settings.rescale_factor)
    normalized = (
        resized.to(dtype=torch.float32)
        .sub(mean.view(-1, 1, 1))
        .div_(std.view(-1, 1, 1))
    )
    patch, merge, temporal = (
        settings.patch_size,
        settings.merge_size,
        settings.temporal_patch_size,
    )
    batch, channel = normalized.shape[:2]
    grid_h, grid_w = height // patch, width // patch
    patches = normalized.reshape(
        batch, channel, grid_h // merge, merge, patch, grid_w // merge, merge, patch
    ).permute(0, 2, 5, 3, 6, 1, 4, 7)
    rows = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
        .reshape(batch, grid_h * grid_w, channel * temporal * patch * patch)
    )
    return ProcessedImage(
        pixel_values=rows[0],
        grid=(1, grid_h, grid_w),
        tokens=grid_h * grid_w // merge**2,
        digest=digest,
    )
