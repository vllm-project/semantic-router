"""Vela Omni audio preprocessing in NumPy: WAV decoding, resampling and log-mel features.

Both branches start from the original PCM: each channel is resampled to
16 kHz (Whisper) and 48 kHz (CLAP) with torchaudio's default Hann-windowed
sinc kernel, then the channels are averaged. Spectrograms follow the
Transformers feature extractors the published model uses: periodic Hann
window, reflect-centred frames, a float64 FFT stored as complex64, Slaney mel
filters (a bundle's exported filters, or the same filters computed from the
published preprocessor configs) and the Whisper or CLAP log readout. WAV
decoding accepts exactly what the router accepts: integer PCM (8, 16, 24, 32
bit) and IEEE float32, up to eight channels and 30 seconds.
"""

from __future__ import annotations

import functools
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

if TYPE_CHECKING:
    from numpy.typing import NDArray

WHISPER_RATE = 16_000
CLAP_RATE = 48_000
CLAP_WINDOW = 10 * CLAP_RATE
MAX_SECONDS = 30
MAX_CHANNELS = 8
MAX_RATE = 384_000
MAX_WAV_BYTES = 32 << 20
LOWPASS_WIDTH = 6
ROLLOFF = 0.99
_EXTENSIBLE = 0xFFFE
_SUBFORMAT_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
_PCM, _FLOAT = 1, 3
_PCM_BITS = (8, 16, 24, 32)
_UNSIGNED_BITS = 8
_FLOAT_BITS = 32
_RIFF_HEADER_BYTES = 12
_CHUNK_HEADER_BYTES = 8
_FILTER_RANK = 2
_FMT_MIN_BYTES = 16
_EXTENSIBLE_MIN_BYTES = 40
_EXTENSIBLE_MIN_EXTRA = 22
# Slaney's mel scale: linear below 1 kHz (15 mels), logarithmic above.
_MIN_LOG_HERTZ = 1000.0
_MIN_LOG_MEL = 15.0


class WhisperSettings(TypedDict):
    feature_extractor_type: str
    feature_size: int
    sampling_rate: int
    n_fft: int
    hop_length: int
    n_samples: int
    nb_max_frames: int
    padding_value: float


class ClapSettings(TypedDict):
    feature_extractor_type: str
    feature_size: int
    sampling_rate: int
    fft_window_size: int
    hop_length: int
    nb_max_samples: int
    frequency_min: int
    frequency_max: int
    padding: str
    truncation: str
    top_db: None


# The published feature extractors' settings (Transformers' WhisperFeatureExtractor
# and ClapFeatureExtractor as Vela 1.0 Omni configures them).
WHISPER_FEATURES: WhisperSettings = {
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 80,
    "sampling_rate": WHISPER_RATE,
    "n_fft": 400,
    "hop_length": 160,
    "n_samples": 480_000,
    "nb_max_frames": 3000,
    "padding_value": 0.0,
}
CLAP_FEATURES: ClapSettings = {
    "feature_extractor_type": "ClapFeatureExtractor",
    "feature_size": 64,
    "sampling_rate": CLAP_RATE,
    "fft_window_size": 1024,
    "hop_length": 480,
    "nb_max_samples": 480_000,
    "frequency_min": 50,
    "frequency_max": 14_000,
    "padding": "repeatpad",
    "truncation": "rand_trunc",
    "top_db": None,
}


def _hertz_to_mel(frequency: float) -> float:
    if frequency >= _MIN_LOG_HERTZ:
        mel: float = _MIN_LOG_MEL + np.log(frequency / _MIN_LOG_HERTZ) * (
            27.0 / np.log(6.4)
        )
        return mel
    return 3.0 * frequency / 200.0


def _mel_to_hertz(mels: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    frequencies = 200.0 * mels / 3.0
    log = mels >= _MIN_LOG_MEL
    frequencies[log] = _MIN_LOG_HERTZ * np.exp(
        np.log(6.4) / 27.0 * (mels[log] - _MIN_LOG_MEL)
    )
    return frequencies


def slaney_filters(
    bins: int, mels: int, low: float, high: float, rate: int
) -> NDArray[np.float64]:
    """Area-normalized triangular filters on Slaney's mel scale, ``[bins, mels]`` float64.

    The operations of Transformers' ``audio_utils.mel_filter_bank`` with
    ``norm="slaney", mel_scale="slaney"``, the filters both published
    feature extractors compute.
    """
    centres = _mel_to_hertz(
        np.linspace(_hertz_to_mel(low), _hertz_to_mel(high), mels + 2)
    )
    spacing = np.diff(centres)
    slopes = np.expand_dims(centres, 0) - np.expand_dims(
        np.linspace(0, rate // 2, bins), 1
    )
    falling = -slopes[:, :-2] / spacing[:-1]
    rising = slopes[:, 2:] / spacing[1:]
    filters: NDArray[np.float64] = np.maximum(np.zeros(1), np.minimum(falling, rising))
    return filters * np.expand_dims(2.0 / (centres[2 : mels + 2] - centres[:mels]), 0)


class AudioError(ValueError):
    """An audio input the model cannot read (reported as ``invalid_input``)."""


@dataclass(frozen=True)
class PCM:
    """Decoded audio: channels-first float32 samples ``[channels, frames]`` at ``rate`` Hz."""

    samples: NDArray[np.float32]
    rate: int

    @property
    def channels(self) -> int:
        channels: int = self.samples.shape[0]
        return channels


@dataclass(frozen=True)
class Spectrum:
    """One log-mel readout of ``processors/audio.json`` (Whisper or CLAP)."""

    sampling_rate: int
    n_fft: int
    hop_length: int
    n_samples: int
    n_frames: int
    mel_filters: NDArray[np.float64]  # [n_fft // 2 + 1, mels]
    clap: bool

    @classmethod
    def from_config(cls, config: dict[str, Any], clap: bool) -> Spectrum:
        expected = (
            {"log": "db", "reference": 1, "min_value": 1e-10, "padding": "repeatpad"}
            if clap
            else {"log": "log10", "range": 8, "affine": [0.25, 1.0]}
        )
        common = {
            "window": "periodic_hann",
            "center": True,
            "pad_mode": "reflect",
            "power": 2,
            "floor": 1e-10,
        }
        if any(
            config.get(key) != value for key, value in {**common, **expected}.items()
        ):
            raise ValueError("unsupported audio feature contract")
        filters = np.asarray(config["mel_filters"], dtype=np.float64)
        n_fft = int(config["n_fft"])
        if (
            filters.ndim != _FILTER_RANK
            or filters.shape[0] != n_fft // 2 + 1
            or (filters < 0).any()
        ):
            raise ValueError("invalid mel filter bank")
        return cls(
            sampling_rate=int(config["sampling_rate"]),
            n_fft=n_fft,
            hop_length=int(config["hop_length"]),
            n_samples=int(config["n_samples"]),
            n_frames=int(config["n_frames"]),
            mel_filters=filters,
            clap=clap,
        )

    @classmethod
    def whisper(cls, preprocessor: dict[str, Any]) -> Spectrum:
        """Whisper's readout from its published ``preprocessor_config.json`` (80 mels up to 8 kHz)."""
        _expect(preprocessor, WHISPER_FEATURES, "Whisper")
        n_fft, mels = WHISPER_FEATURES["n_fft"], WHISPER_FEATURES["feature_size"]
        return cls(
            sampling_rate=WHISPER_RATE,
            n_fft=n_fft,
            hop_length=WHISPER_FEATURES["hop_length"],
            n_samples=WHISPER_FEATURES["n_samples"],
            n_frames=WHISPER_FEATURES["nb_max_frames"],
            mel_filters=slaney_filters(n_fft // 2 + 1, mels, 0.0, 8000.0, WHISPER_RATE),
            clap=False,
        )

    @classmethod
    def clap_window(cls, preprocessor: dict[str, Any]) -> Spectrum:
        """CLAP's readout of one 10-second window from its published ``preprocessor_config.json``."""
        _expect(preprocessor, CLAP_FEATURES, "CLAP")
        n_fft, hop = CLAP_FEATURES["fft_window_size"], CLAP_FEATURES["hop_length"]
        samples = CLAP_FEATURES["nb_max_samples"]
        return cls(
            sampling_rate=CLAP_RATE,
            n_fft=n_fft,
            hop_length=hop,
            n_samples=samples,
            n_frames=samples // hop + 1,
            mel_filters=slaney_filters(
                n_fft // 2 + 1,
                CLAP_FEATURES["feature_size"],
                CLAP_FEATURES["frequency_min"],
                CLAP_FEATURES["frequency_max"],
                CLAP_RATE,
            ),
            clap=True,
        )


def _expect(config: dict[str, Any], expected: Mapping[str, object], name: str) -> None:
    changed = sorted(key for key, value in expected.items() if config.get(key) != value)
    if changed:
        raise ValueError(f"unsupported {name} feature extractor settings: {changed}")


def decode_wav(data: bytes) -> PCM:
    """Integer PCM or float32 WAV to channels-first float32, exactly as the router decodes it."""
    if (
        len(data) < _RIFF_HEADER_BYTES
        or len(data) > MAX_WAV_BYTES
        or data[:4] != b"RIFF"
        or data[8:12] != b"WAVE"
    ):
        raise AudioError("missing RIFF/WAVE header or oversized payload")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise AudioError("RIFF size does not match payload")
    fmt: tuple[int, int, int, int, int] | None = None
    samples: bytes | None = None
    offset = _RIFF_HEADER_BYTES
    while offset < len(data):
        if len(data) - offset < _CHUNK_HEADER_BYTES:
            raise AudioError("incomplete chunk header")
        kind, size = (
            data[offset : offset + 4],
            struct.unpack_from("<I", data, offset + 4)[0],
        )
        start, end = offset + 8, offset + 8 + size
        if size > MAX_WAV_BYTES or end > len(data):
            raise AudioError("chunk extends beyond payload")
        if kind == b"fmt ":
            if fmt is not None or size < _FMT_MIN_BYTES:
                raise AudioError("duplicate or incomplete format")
            code, channels, rate, _, block, bits = struct.unpack_from(
                "<HHIIHH", data, start
            )
            if code == _EXTENSIBLE:
                extra, valid = struct.unpack_from("<HH", data, start + 16)
                if (
                    size < _EXTENSIBLE_MIN_BYTES
                    or extra < _EXTENSIBLE_MIN_EXTRA
                    or valid != bits
                    or data[start + 26 : start + 40] != _SUBFORMAT_TAIL
                ):
                    raise AudioError("unsupported extensible format")
                code = struct.unpack_from("<H", data, start + 24)[0]
            fmt = (code, channels, rate, block, bits)
        elif kind == b"data":
            if samples is not None:
                raise AudioError("multiple data chunks are unsupported")
            samples = data[start:end]
        offset = end + (size & 1)
        if offset > len(data):
            raise AudioError("missing chunk padding")
    if fmt is None or not samples:
        raise AudioError("missing format or samples")
    code, channels, rate, block, bits = fmt
    if not 1 <= channels <= MAX_CHANNELS or not 1 <= rate <= MAX_RATE:
        raise AudioError("invalid channels or rate")
    if not (
        (code == _PCM and bits in _PCM_BITS) or (code == _FLOAT and bits == _FLOAT_BITS)
    ):
        raise AudioError("only integer PCM and IEEE float32 are supported")
    width = bits // 8
    if block != channels * width or len(samples) % block:
        raise AudioError("invalid frame alignment")
    frames = len(samples) // block
    if frames > MAX_SECONDS * rate:
        raise AudioError("duration exceeds 30 seconds")
    raw = np.frombuffer(samples, dtype=np.uint8).reshape(frames, channels, width)
    if code == _FLOAT:
        values = raw.copy().view("<f4")[..., 0]
    elif bits == _UNSIGNED_BITS:
        values = (raw[..., 0].astype(np.int32) - 128).astype(np.float32) / np.float32(
            128
        )
    else:
        padded = np.zeros((frames, channels, 4), dtype=np.uint8)
        padded[..., 4 - width :] = raw
        signed = padded.view("<i4")[..., 0] >> (32 - bits)
        values = signed.astype(np.float32) / np.float32(1 << (bits - 1))
    pcm = np.ascontiguousarray(values.T, dtype=np.float32)
    if not np.isfinite(pcm).all():
        raise AudioError("samples must be finite")
    return PCM(pcm, rate)


@functools.lru_cache(maxsize=32)
def _kernel(original: int, new: int) -> tuple[NDArray[np.float64], int]:
    """torchaudio's Hann sinc kernel for coprime rates, transposed (``[2 * width + original, new]``, float64).

    It depends only on the rates, so each pair is built once (read-only).
    """
    base = min(original, new) * ROLLOFF
    width = math.ceil(LOWPASS_WIDTH * original / base)
    # The phase is divided in float32 (torch.arange's default dtype), the rest in float64.
    phase = (-np.arange(new, dtype=np.float32) / np.float32(new)).astype(np.float64)
    offsets = np.arange(-width, width + original, dtype=np.float64) / original
    t = np.clip(
        (phase[:, None] + offsets[None, :]) * base, -LOWPASS_WIDTH, LOWPASS_WIDTH
    )
    window = np.cos(t * math.pi / LOWPASS_WIDTH / 2) ** 2
    angle = t * math.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(angle == 0, 1.0, np.sin(angle) / angle)
    kernel = (sinc * window * (base / original)).astype(np.float32).T.astype(np.float64)
    kernel.setflags(write=False)
    return kernel, width


def resample(
    signal: NDArray[np.float32], source: int, target: int
) -> NDArray[np.float32]:
    """One channel from ``source`` Hz to ``target`` Hz; the output has ``ceil(n * target / source)`` samples."""
    if source == target:
        return signal.astype(np.float32, copy=True)
    divisor = math.gcd(source, target)
    original, new = source // divisor, target // divisor
    kernel, width = _kernel(original, new)
    taps = kernel.shape[0]
    length = -(-len(signal) * new // original)
    groups = -(-length // new)
    right = max((groups - 1) * original + taps - width - len(signal), 0)
    padded = np.pad(signal.astype(np.float64), (width, right))
    frames = sliding_window_view(padded, taps)[::original][:groups]
    out: NDArray[np.float64] = frames @ kernel
    return out.reshape(-1)[:length].astype(np.float32)


def native_rate(pcm: PCM, target: int) -> NDArray[np.float32]:
    """Every channel resampled to ``target`` and averaged; at most 30 seconds."""
    length = min(-(-pcm.samples.shape[1] * target // pcm.rate), target * MAX_SECONDS)
    total = np.zeros(length, dtype=np.float32)
    for channel in pcm.samples:
        values = resample(channel, pcm.rate, target)[:length]
        total[: len(values)] += values
    mean: NDArray[np.float32] = total / np.float32(pcm.channels)
    return mean


def windows(samples: int) -> list[tuple[int, int]]:
    """The published 48 kHz endpoint-cover segmentation; the tail is never dropped."""
    if samples <= CLAP_WINDOW:
        return [(0, samples)]
    count, last = -(-samples // CLAP_WINDOW), samples - CLAP_WINDOW
    return [
        (index * last // (count - 1), index * last // (count - 1) + CLAP_WINDOW)
        for index in range(count)
    ]


def features(wave: NDArray[np.float32], spec: Spectrum) -> NDArray[np.float32]:
    """Log-mel features: Whisper ``[mels, frames]`` normalized; CLAP ``[frames, mels]`` in dB."""
    padded = np.zeros(spec.n_samples, dtype=np.float32)
    if spec.clap:
        repeated = (spec.n_samples // len(wave)) * len(wave)
        padded[:repeated] = np.tile(wave, spec.n_samples // len(wave))
    else:
        padded[: min(len(wave), spec.n_samples)] = wave[: spec.n_samples]
    half = spec.n_fft // 2
    centred = np.pad(padded.astype(np.float64), (half, half), mode="reflect")
    # Whisper zero-pads to 30 s; frames past the signal hold only zeros, so their
    # power is exactly 0 and their energy exactly the floor.
    live = spec.n_frames
    if not spec.clap:
        live = min(live, -(-(min(len(wave), spec.n_samples) + half) // spec.hop_length))
    frames = sliding_window_view(centred, spec.n_fft)[:: spec.hop_length][:live]
    window = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(spec.n_fft) / spec.n_fft)
    spectrum = np.fft.rfft(frames * window, axis=1).astype(np.complex64)
    power = (
        spectrum.real.astype(np.float64) ** 2 + spectrum.imag.astype(np.float64) ** 2
    )
    energy = np.full((spec.n_frames, spec.mel_filters.shape[1]), 1e-10)
    energy[:live] = np.maximum(power @ spec.mel_filters, 1e-10)
    if spec.clap:
        decibels: NDArray[np.float32] = (10.0 * np.log10(energy)).astype(np.float32)
        return decibels
    log = np.log10(energy).T.astype(np.float32)
    normalized: NDArray[np.float32] = (
        np.maximum(log, log.max() - np.float32(8.0)) + np.float32(4.0)
    ) / np.float32(4.0)
    return normalized


def validate(pcm: PCM, sample_rates: tuple[int, ...], max_sample_rate: int) -> None:
    """The bundle's input contract: an allowed rate, 1-8 finite channels, at most 30 seconds."""
    if not 0 < pcm.rate <= max_sample_rate or (
        sample_rates and pcm.rate not in sample_rates
    ):
        raise AudioError("unsupported original audio sampling rate")
    if not 1 <= pcm.channels <= MAX_CHANNELS or pcm.samples.shape[1] == 0:
        raise AudioError("require 1-8 nonempty channels")
    if pcm.samples.shape[1] > pcm.rate * MAX_SECONDS:
        raise AudioError("audio input exceeds 30 seconds")
