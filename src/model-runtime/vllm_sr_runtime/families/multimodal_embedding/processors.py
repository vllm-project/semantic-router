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
from ...heads import embedding
from . import audio
from .bundle import OmniBundle

MAX_IMAGE_PIXELS = 64 << 20
_RESAMPLE = {"bicubic": "BICUBIC", "bilinear": "BILINEAR"}
# Prefixes per input_type of each instruction API a bundle may declare (Mini's retrieval task).
INSTRUCTIONS = {
    "qwen_optional_instruction_v1": {
        "query": "Instruct: Given a search query, retrieve relevant passages that answer "
        "the query.\nQuery:",
        "document": "",
    }
}


@dataclass(frozen=True)
class AudioFeatures:
    """Graph inputs of one audio input: CLAP windows ``[n, 1, 1, 1001, 64]`` and Whisper ``[1, 80, 3000]``."""

    clap: np.ndarray
    whisper: np.ndarray


class TextProcessor:
    """The bundle tokenizer with its special tokens; over-long input is rejected or, under ``truncate``, cut.

    A bundle with an instruction API formats the raw text for its ``input_type``
    first (a query gets the task instruction), then strips the result. A cut
    keeps the beginning of the content inside the special tokens
    (``heads.embedding.encode_text``).
    """

    def __init__(self, bundle: OmniBundle):
        from tokenizers import Tokenizer

        settings = bundle.processors["text"]
        self.backend = Tokenizer.from_file(
            str(bundle.file(bundle.manifest["tokenizer"]))
        )
        self.backend.no_truncation()
        self.backend.no_padding()
        self.strip = bool(settings["strip_whitespace"])
        self.instructions = INSTRUCTIONS.get(settings["instruction_api"] or "", {})
        self.pad_id = int(settings["pad_token_id"])
        if self.pad_id >= self.backend.get_vocab_size(with_added_tokens=True):
            raise ValueError("the pad token is outside the tokenizer vocabulary")

    @property
    def input_types(self) -> tuple[str, ...]:
        return tuple(self.instructions)

    def encode(
        self,
        text: str,
        budget: int,
        input_type: str | None = None,
        overflow: str = "reject",
    ) -> tuple[list[int], dict[str, Any]] | str:
        if input_type is not None:
            text = self.instructions[input_type] + text
        encoded = embedding.encode_text(
            self.backend, text.strip() if self.strip else text, budget, overflow
        )
        if not isinstance(encoded, str) and not encoded[0]:
            return MAX_LENGTH_EXCEEDED
        return encoded


class ImageProcessor:
    """Decoded RGB resized to the graph's square input, rescaled and normalized, channels first.

    A pixel's normalized value depends only on its byte and channel, so the
    rescale and normalization are one table of 256 values per channel,
    computed with the processor's own float64 and float32 steps.
    """

    def __init__(self, bundle: OmniBundle):
        settings = bundle.processors["image"]
        self.size = int(settings["size"])
        self.resample = _RESAMPLE[settings["resample"]]
        self.mean = np.asarray(settings["mean"], dtype=np.float32)
        self.std = np.asarray(settings["std"], dtype=np.float32)
        scaled = (np.arange(256, dtype=np.float64) * (1 / 255)).astype(np.float32)
        self.table = np.ascontiguousarray(((scaled[:, None] - self.mean) / self.std).T)

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
        resized = np.asarray(
            rgb.resize((self.size, self.size), resample=resample, reducing_gap=None)
        )
        pixels = np.empty((1, 3, self.size, self.size), dtype=np.float32)
        for channel in range(3):
            # Indexing gathers twice as fast as np.take(..., out=) on strided channels.
            pixels[0, channel] = self.table[channel][resized[..., channel]]
        return pixels


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
