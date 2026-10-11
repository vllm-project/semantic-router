"""Decision 3.0 video inputs: decoding, frame sampling and the Qwen3-VL video processor, as the d3 runtime reads them.

A request's videos are base64 MP4, WebM, QuickTime or Matroska data URLs of at
most 32,000,000 bytes, 300 seconds and 8,294,400 pixels per frame (the d3
server's strict loading). OpenCV decodes each from a temporary file with its
FFmpeg backend: 2 frames per second, at least 4 and at most 32, spread evenly
over the whole video (a frame rate the container does not give plausibly is 24
per second), converted from BGR to RGB; an odd count repeats the last frame.
The processor is the Transformers 5.17 ``Qwen3VLVideoProcessor`` torchvision
backend with the runtime's budget: at most 200,704 pixels per frame
(``cap_pixels_per_frame``), both sides a multiple of ``patch_size *
merge_size``, torchvision's uint8 bicubic antialiased resize, rescale and
normalize fused into one FP32 subtraction and division, then the patch layout
over each pair of frames. Every pair of frames is one group of input tokens
after its timestamp; the videos of a request take at most 16,384 input tokens.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import torch

from .images import _PREPROCESS, _resize

FPS = 2.0
MIN_FRAMES = 4
MAX_FRAMES = 32
# Per frame; every two frames take one input token per 32 x 32 pixels.
MAX_PIXELS = 200_704
MIN_PIXELS = 4_096
# All videos of one request.
MAX_TOKENS = 16_384
MAX_VIDEO_BYTES = 32_000_000
MAX_SOURCE_PIXELS = 8_294_400
MAX_SECONDS = 300
DATA_URL_FORMATS = ("mp4", "webm", "quicktime", "x-matroska")
# A source frame rate outside this range is treated as unknown.
SOURCE_FPS = (0.1, 1000.0)
UNKNOWN_FPS = 24.0
MAX_ASPECT_RATIO = 200


def available() -> str | None:
    """Why videos cannot be decoded here (OpenCV is missing), or None."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return "decoding videos needs OpenCV (opencv-python-headless)"
    return None


@dataclass(frozen=True)
class VideoSettings:
    """The package's video processor settings (``video_preprocessor_config.json``) with the runtime's budget."""

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    rescale_factor: float = 1 / 255

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VideoSettings:
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
class Video:
    """A decoded video: its sampled RGB frames (uint8 ``[frames, height, width, 3]``, an even count) and their sources."""

    frames: Any
    fps: float
    indices: tuple[int, ...]
    total: int
    digest: str


@dataclass(frozen=True)
class ProcessedVideo:
    """One video as the vision tower reads it: FP32 patch rows, its (t, h, w) grid, tokens and frame-pair timestamps."""

    pixel_values: torch.Tensor
    grid: tuple[int, int, int]
    tokens: int
    timestamps: tuple[float, ...]
    digest: str

    def placeholder(self, start: str, pad: str, end: str) -> str:
        """The processor's expansion of this video's placeholder: each frame pair after its timestamp."""
        group = self.tokens // self.grid[0]
        return "".join(
            f"<{seconds:.1f} seconds>" + start + pad * group + end
            for seconds in self.timestamps
        )


def sample_indices(total: int, fps: float) -> list[int]:
    """Source frames to read: 2 per second, at least 4 and at most 32, spread evenly over the whole video."""
    import numpy as np

    count = int(total / fps * FPS)
    count = min(max(count, MIN_FRAMES), MAX_FRAMES, total)
    return [int(i) for i in np.linspace(0, total - 1, count).round().astype(int)]


def _payload(value: Any) -> bytes:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise ValueError(
            "videos must be base64 MP4, WebM, QuickTime or Matroska data URLs"
        )
    header, separator, encoded = value.partition(",")
    kind = header.strip().lower()
    if (
        not separator
        or not kind.startswith("data:video/")
        or not kind.endswith(";base64")
    ):
        raise ValueError("a data URL video is data:video/<format>;base64,<data>")
    if kind[len("data:video/") :].split(";")[0] not in DATA_URL_FORMATS:
        raise ValueError(
            "videos must be base64 MP4, WebM, QuickTime or Matroska data URLs"
        )
    if len(encoded) > 4 * -(-MAX_VIDEO_BYTES // 3):
        raise ValueError(f"each video must be at most {MAX_VIDEO_BYTES:,} bytes")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 video data") from exc
    if len(payload) > MAX_VIDEO_BYTES:
        raise ValueError(f"each video must be at most {MAX_VIDEO_BYTES:,} bytes")
    return payload


def decode(value: Any) -> Video:
    """A video data URL decoded and sampled as the d3 runtime does it; ValueError when invalid."""
    try:
        import cv2
    except ImportError as exc:
        raise ValueError(
            "decoding videos needs OpenCV (opencv-python-headless)"
        ) from exc
    payload = _payload(value)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        api = (
            cv2.CAP_FFMPEG
            if cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG)
            else cv2.CAP_ANY
        )
    except (AttributeError, cv2.error):
        api = cv2.CAP_ANY
    # Encoded bytes are read from a file, as the d3 runtime reads them.
    with tempfile.NamedTemporaryFile(prefix="vllm-srun-video-", delete=False) as handle:
        handle.write(payload)
        path = handle.name
    frames: list[Any] = []
    kept: list[int] = []
    reader = None
    try:
        reader = cv2.VideoCapture(path, api)
        if not reader.isOpened():
            raise ValueError("the video could not be opened")
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("the video has no frame size")
        if width * height > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"video frames must have at most {MAX_SOURCE_PIXELS:,} pixels"
            )
        fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (math.isfinite(fps) and SOURCE_FPS[0] <= fps <= SOURCE_FPS[1]):
            fps = UNKNOWN_FPS
        total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:  # the container does not say: count the frames, then read again
            limit = int(MAX_SECONDS * fps)
            while reader.grab():
                total += 1
                if total > limit:
                    break
            reader.release()
            reader = cv2.VideoCapture(path, api)
        if total <= 0:
            raise ValueError("the video has no frames")
        if total / fps > MAX_SECONDS:
            raise ValueError(f"each video must be at most {MAX_SECONDS} seconds long")
        indices = sample_indices(total, fps)
        wanted = set(indices)
        for number in range(indices[-1] + 1):
            if not reader.grab():
                continue
            if number in wanted:
                ok, frame = reader.retrieve()
                if ok and frame is not None and frame.size:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    kept.append(number)
    except cv2.error as exc:
        raise ValueError(f"invalid video data ({exc.__class__.__name__})") from exc
    finally:
        if reader is not None:
            reader.release()
        os.unlink(path)
    if not frames:
        raise ValueError("no frame of the video could be decoded")
    return _video(frames, kept, fps, total, digest)


def _video(
    frames: list[Any], indices: list[int], fps: float, total: int, digest: str
) -> Video:
    import numpy as np

    if len(frames) % 2:
        frames.append(frames[-1])
        indices.append(indices[-1])
    try:
        stacked = np.stack(frames)
    except ValueError as exc:
        raise ValueError("all frames of a video must have the same size") from exc
    return Video(
        np.ascontiguousarray(stacked), float(fps), tuple(indices), int(total), digest
    )


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """The Qwen3-VL video ``smart_resize``: both sides a multiple of ``factor``, the frames' area within the budget."""
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def frame_size(
    settings: VideoSettings, frames: int, height: int, width: int
) -> tuple[int, int]:
    """The resized (height, width) of a video's frames: the per-frame cap of ``cap_pixels_per_frame``."""
    factor = settings.factor
    budget = MAX_PIXELS * MAX_FRAMES
    per_frame = max(
        min(MAX_PIXELS // factor**2 * factor**2, budget // frames),
        int(MIN_PIXELS * 1.05),
    )
    return smart_resize(
        frames,
        height,
        width,
        temporal_factor=settings.temporal_patch_size,
        factor=factor,
        min_pixels=MIN_PIXELS,
        max_pixels=per_frame * frames,
    )


def input_tokens(settings: VideoSettings, frames: int, height: int, width: int) -> int:
    """The input tokens of one video of this many frames of this size (timestamps not counted)."""
    resized_height, resized_width = frame_size(settings, frames, height, width)
    grid = (
        frames // settings.temporal_patch_size,
        resized_height // settings.patch_size,
        resized_width // settings.patch_size,
    )
    return grid[0] * grid[1] * grid[2] // settings.merge_size**2


def timestamps(indices: tuple[int, ...], fps: float, merge: int) -> tuple[float, ...]:
    """The Qwen3-VL processor's timestamp of each group of ``merge`` frames: the mean of its first and last frame."""
    seconds = [index / fps for index in indices]
    return tuple(
        (seconds[i] + seconds[i + merge - 1]) / 2 for i in range(0, len(seconds), merge)
    )


def preprocess(video: Video, settings: VideoSettings) -> ProcessedVideo:
    """One decoded video through the processor (on the preprocessing thread): FP32 patch rows and its grid."""
    return _PREPROCESS.submit(_preprocess, video, settings).result()


def _preprocess(video: Video, settings: VideoSettings) -> ProcessedVideo:
    frames = torch.from_numpy(video.frames).contiguous()
    frames = frames.permute(0, 3, 1, 2).contiguous()
    stacked = torch.stack([frames])
    count, _, old_height, old_width = frames.shape
    height, width = frame_size(settings, count, old_height, old_width)
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
    batch, _, channel = normalized.shape[:3]
    grid_t, grid_h, grid_w = count // temporal, height // patch, width // patch
    rows = (
        normalized.view(
            batch,
            grid_t,
            temporal,
            channel,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        .reshape(batch, grid_t * grid_h * grid_w, channel * temporal * patch * patch)
    )
    return ProcessedVideo(
        pixel_values=rows[0],
        grid=(grid_t, grid_h, grid_w),
        tokens=grid_t * grid_h * grid_w // merge**2,
        timestamps=timestamps(video.indices, video.fps, temporal),
        digest=video.digest,
    )
