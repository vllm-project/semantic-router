"""Vela Omni input processors: text tokens, image pixels and audio features as model inputs.

Every processor reproduces the Transformers processor of the published
model (the one a prepared bundle was exported and parity-checked with).
Images decode and resize through Pillow (the processors' own backend:
antialiased resampling with 8-bit intermediates), then rescale in float64 and
normalize in float32. Audio follows ``audio.py``. Each processor is built
from a bundle's manifest or from the published repository's configs.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...errors import INVALID_INPUT, MAX_LENGTH_EXCEEDED
from ...heads import embedding
from . import audio
from .bundle import OmniBundle

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_IMAGE_PIXELS = 64 << 20
_RESAMPLE = {"bicubic": "BICUBIC", "bilinear": "BILINEAR"}
# PIL resampling codes in an image processor config.
RESAMPLE_CODES = {3: "bicubic", 2: "bilinear"}
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

    clap: NDArray[np.float32]
    whisper: NDArray[np.float32]


class TextProcessor:
    """The model's tokenizer with its special tokens; over-long input is rejected or, under ``truncate``, cut.

    A model with an instruction API formats the raw text for its ``input_type``
    first (a query gets the task instruction), then strips the result. A cut
    keeps the beginning of the content inside the special tokens
    (``heads.embedding.encode_text``).
    """

    def __init__(
        self,
        tokenizer: Path,
        *,
        strip_whitespace: bool,
        instruction_api: str | None,
        pad_token_id: int | None = None,
    ):
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_file(str(tokenizer))
        self.backend.no_truncation()
        self.backend.no_padding()
        self.strip = strip_whitespace
        self.instructions = INSTRUCTIONS.get(instruction_api or "", {})
        vocabulary = self.backend.get_vocab_size(with_added_tokens=True)
        if pad_token_id is not None and pad_token_id >= vocabulary:
            raise ValueError("the pad token is outside the tokenizer vocabulary")

    @classmethod
    def from_bundle(cls, bundle: OmniBundle) -> TextProcessor:
        settings = bundle.processors["text"]
        return cls(
            bundle.file(bundle.manifest["tokenizer"]),
            strip_whitespace=bool(settings["strip_whitespace"]),
            instruction_api=settings["instruction_api"],
            pad_token_id=int(settings["pad_token_id"]),
        )

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
    computed with the processor's own float64 and float32 steps. Pillow
    resamples each band of an RGB image with the same integer arithmetic as a
    one-band image, so the three bands resize and normalize at once, each on
    its own thread and into its own plane, and the pixels are those of the
    whole image's resize.
    """

    def __init__(self, size: int, resample: str, mean: list[float], std: list[float]):
        self.size = size
        self.resample = _RESAMPLE[resample]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        scaled = (np.arange(256, dtype=np.float64) * (1 / 255)).astype(np.float32)
        self.table = np.ascontiguousarray(((scaled[:, None] - self.mean) / self.std).T)
        self.bands = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="vllm-sr-omni-pixels"
        )

    @classmethod
    def from_bundle(cls, bundle: OmniBundle) -> ImageProcessor:
        settings = bundle.processors["image"]
        return cls(
            int(settings["size"]),
            settings["resample"],
            settings["mean"],
            settings["std"],
        )

    @classmethod
    def from_preprocessor(cls, config: dict[str, Any]) -> ImageProcessor:
        """A published ``SiglipImageProcessor`` config: a square resize, rescale by 1/255, normalize."""
        size = config.get("size") or {}
        if (
            config.get("image_processor_type") != "SiglipImageProcessor"
            or not (config.get("do_resize") and config.get("do_rescale"))
            or not config.get("do_normalize")
            or config.get("rescale_factor") != 1 / 255
            or set(size) != {"height", "width"}
            or size["height"] != size["width"]
            or config.get("resample") not in RESAMPLE_CODES
        ):
            raise ValueError("unsupported image processor settings")
        return cls(
            int(size["height"]),
            RESAMPLE_CODES[config["resample"]],
            config["image_mean"],
            config["image_std"],
        )

    def pixels(self, data: bytes) -> NDArray[np.float32] | str:
        """``[1, 3, size, size]`` float32, or ``invalid_input`` for an unreadable image."""
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    return INVALID_INPUT
                image.load()
                rgb = image if image.mode == "RGB" else image.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            return INVALID_INPUT
        resample = getattr(Image.Resampling, self.resample)
        bands = rgb.split()
        pixels = np.empty((1, 3, self.size, self.size), dtype=np.float32)

        def channel(index: int) -> None:
            resized = bands[index].resize(
                (self.size, self.size), resample=resample, reducing_gap=None
            )
            pixels[0, index] = self.table[index][np.asarray(resized)]

        others = [self.bands.submit(channel, index) for index in (1, 2)]
        channel(0)
        for other in others:
            other.result()
        return pixels

    def close(self) -> None:
        self.bands.shutdown(wait=True)


class AudioProcessor:
    """WAV bytes to the CLAP and Whisper inputs of the audio branch."""

    def __init__(
        self,
        sample_rates: tuple[int, ...],
        max_sample_rate: int,
        whisper: audio.Spectrum,
        clap: audio.Spectrum,
    ):
        self.sample_rates = sample_rates
        self.max_sample_rate = max_sample_rate
        self.whisper = whisper
        self.clap = clap

    @classmethod
    def from_bundle(cls, bundle: OmniBundle, config: dict[str, Any]) -> AudioProcessor:
        settings = bundle.processors["audio"]
        if (
            config.get("format_version") != 1
            or config.get("windows") != "endpoint_cover_v1"
        ):
            raise ValueError("unsupported audio feature contract")
        return cls(
            tuple(int(rate) for rate in settings["sample_rates"]),
            int(settings["max_sample_rate"]),
            audio.Spectrum.from_config(config["whisper"], clap=False),
            audio.Spectrum.from_config(config["clap"], clap=True),
        )

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
