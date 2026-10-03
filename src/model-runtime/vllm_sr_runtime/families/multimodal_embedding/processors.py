"""Vela Omni input processors: text tokens, image pixels and audio features as graph inputs.

Every processor reproduces the Transformers processor the bundle was exported
and parity-checked with. Images decode and resize through Pillow (the
processors' own backend: antialiased resampling with 8-bit intermediates),
then rescale in float64 and normalize in float32. Audio follows ``audio.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...errors import INVALID_INPUT, MAX_LENGTH_EXCEEDED
from . import audio
from .bundle import OmniBundle

MAX_IMAGE_PIXELS = 64 << 20
_RESAMPLE = {"bicubic": "BICUBIC", "bilinear": "BILINEAR"}


@dataclass(frozen=True)
class AudioFeatures:
    """Graph inputs of one audio input: CLAP windows ``[n, 1, 1, 1001, 64]`` and Whisper ``[1, 80, 3000]``."""

    clap: np.ndarray
    whisper: np.ndarray


class TextProcessor:
    """The bundle tokenizer with its special tokens; over-long input is rejected, never cut."""

    def __init__(self, bundle: OmniBundle):
        from tokenizers import Tokenizer

        settings = bundle.processors["text"]
        self.backend = Tokenizer.from_file(
            str(bundle.file(bundle.manifest["tokenizer"]))
        )
        self.backend.no_truncation()
        self.backend.no_padding()
        self.strip = bool(settings["strip_whitespace"])
        self.pad_id = int(settings["pad_token_id"])
        if self.pad_id >= self.backend.get_vocab_size(with_added_tokens=True):
            raise ValueError("the pad token is outside the tokenizer vocabulary")

    def encode(self, text: str, budget: int) -> tuple[list[int], dict[str, Any]] | str:
        ids = list(self.backend.encode(text.strip() if self.strip else text).ids)
        if not ids or len(ids) > budget:
            return MAX_LENGTH_EXCEEDED
        return ids, {
            "tokens": len(ids),
            "processed_tokens": len(ids),
            "truncated": False,
        }


class ImageProcessor:
    """Decoded RGB resized to the graph's square input, rescaled and normalized, channels first."""

    def __init__(self, bundle: OmniBundle):
        settings = bundle.processors["image"]
        self.size = int(settings["size"])
        self.resample = _RESAMPLE[settings["resample"]]
        self.mean = np.asarray(settings["mean"], dtype=np.float32)
        self.std = np.asarray(settings["std"], dtype=np.float32)

    def pixels(self, data: bytes) -> np.ndarray | str:
        """``[1, 3, size, size]`` float32, or ``invalid_input`` for an unreadable image."""
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    return INVALID_INPUT
                rgb = image.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            return INVALID_INPUT
        resample = getattr(Image.Resampling, self.resample)
        resized = rgb.resize(
            (self.size, self.size), resample=resample, reducing_gap=None
        )
        scaled = (np.asarray(resized, dtype=np.float64) * (1 / 255)).astype(np.float32)
        normalized = (scaled - self.mean) / self.std
        return np.ascontiguousarray(
            normalized.transpose(2, 0, 1)[None], dtype=np.float32
        )


class AudioProcessor:
    """WAV bytes to the CLAP and Whisper inputs of the audio branch."""

    def __init__(self, bundle: OmniBundle, config: dict[str, Any]):
        settings = bundle.processors["audio"]
        self.sample_rates = tuple(int(rate) for rate in settings["sample_rates"])
        self.max_sample_rate = int(settings["max_sample_rate"])
        if (
            config.get("format_version") != 1
            or config.get("windows") != "endpoint_cover_v1"
        ):
            raise ValueError("unsupported audio feature contract")
        self.whisper = audio.Spectrum.from_config(config["whisper"], clap=False)
        self.clap = audio.Spectrum.from_config(config["clap"], clap=True)

    def features(self, data: bytes, media_type: str | None) -> AudioFeatures | str:
        if media_type not in (
            None,
            "wav",
            "wave",
            "audio/wav",
            "audio/wave",
            "audio/x-wav",
        ):
            return INVALID_INPUT
        try:
            pcm = audio.decode_wav(data)
            audio.validate(pcm, self.sample_rates, self.max_sample_rate)
        except audio.AudioError:
            return INVALID_INPUT
        speech = audio.native_rate(pcm, audio.WHISPER_RATE)
        wide = audio.native_rate(pcm, audio.CLAP_RATE)
        clap = np.stack(
            [
                audio.features(wide[start:end], self.clap)
                for start, end in audio.windows(len(wide))
            ]
        )
        whisper = audio.features(speech, self.whisper)
        return AudioFeatures(
            clap=np.ascontiguousarray(clap[:, None, None], dtype=np.float32),
            whisper=np.ascontiguousarray(whisper[None], dtype=np.float32),
        )
