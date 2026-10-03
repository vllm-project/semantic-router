"""Vela Omni audio preprocessing in NumPy: WAV decoding, resampling and log-mel features.

Both branches start from the original PCM: each channel is resampled to
16 kHz (Whisper) and 48 kHz (CLAP) with torchaudio's default Hann-windowed
sinc kernel, then the channels are averaged. Spectrograms follow the
Transformers feature extractors the bundle was exported with: periodic Hann
window, reflect-centred frames, a float64 FFT stored as complex64, the
bundle's mel filters, and the Whisper or CLAP log readout. WAV decoding
accepts exactly what the router accepts: integer PCM (8, 16, 24, 32 bit) and
IEEE float32, up to eight channels and 30 seconds.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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


class AudioError(ValueError):
    """An audio input the model cannot read (reported as ``invalid_input``)."""


@dataclass(frozen=True)
class PCM:
    """Decoded audio: channels-first float32 samples ``[channels, frames]`` at ``rate`` Hz."""

    samples: np.ndarray
    rate: int

    @property
    def channels(self) -> int:
        return self.samples.shape[0]


@dataclass(frozen=True)
class Spectrum:
    """One log-mel readout of ``processors/audio.json`` (Whisper or CLAP)."""

    sampling_rate: int
    n_fft: int
    hop_length: int
    n_samples: int
    n_frames: int
    mel_filters: np.ndarray  # [n_fft // 2 + 1, mels], float64
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


def _kernel(original: int, new: int) -> tuple[np.ndarray, int]:
    """torchaudio's Hann sinc kernel ``[new, 2 * width + original]`` for coprime rates."""
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
    return (sinc * window * (base / original)).astype(np.float32), width


def resample(signal: np.ndarray, source: int, target: int) -> np.ndarray:
    """One channel from ``source`` Hz to ``target`` Hz; the output has ``ceil(n * target / source)`` samples."""
    if source == target:
        return signal.astype(np.float32, copy=True)
    divisor = math.gcd(source, target)
    original, new = source // divisor, target // divisor
    kernel, width = _kernel(original, new)
    length = -(-len(signal) * new // original)
    groups = -(-length // new)
    right = max((groups - 1) * original + kernel.shape[1] - width - len(signal), 0)
    padded = np.pad(signal.astype(np.float64), (width, right))
    frames = sliding_window_view(padded, kernel.shape[1])[::original][:groups]
    out = frames @ kernel.T.astype(np.float64)
    return out.reshape(-1)[:length].astype(np.float32)


def native_rate(pcm: PCM, target: int) -> np.ndarray:
    """Every channel resampled to ``target`` and averaged; at most 30 seconds."""
    length = min(-(-pcm.samples.shape[1] * target // pcm.rate), target * MAX_SECONDS)
    total = np.zeros(length, dtype=np.float32)
    for channel in pcm.samples:
        values = resample(channel, pcm.rate, target)[:length]
        total[: len(values)] += values
    return total / np.float32(pcm.channels)


def windows(samples: int) -> list[tuple[int, int]]:
    """The published 48 kHz endpoint-cover segmentation; the tail is never dropped."""
    if samples <= CLAP_WINDOW:
        return [(0, samples)]
    count, last = -(-samples // CLAP_WINDOW), samples - CLAP_WINDOW
    return [
        (index * last // (count - 1), index * last // (count - 1) + CLAP_WINDOW)
        for index in range(count)
    ]


def features(wave: np.ndarray, spec: Spectrum) -> np.ndarray:
    """Log-mel features: Whisper ``[mels, frames]`` normalized; CLAP ``[frames, mels]`` in dB."""
    padded = np.zeros(spec.n_samples, dtype=np.float32)
    if spec.clap:
        repeated = (spec.n_samples // len(wave)) * len(wave)
        padded[:repeated] = np.tile(wave, spec.n_samples // len(wave))
    else:
        padded[: min(len(wave), spec.n_samples)] = wave[: spec.n_samples]
    half = spec.n_fft // 2
    centred = np.pad(padded.astype(np.float64), (half, half), mode="reflect")
    frames = sliding_window_view(centred, spec.n_fft)[:: spec.hop_length][
        : spec.n_frames
    ]
    window = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(spec.n_fft) / spec.n_fft)
    spectrum = np.fft.rfft(frames * window, axis=1).astype(np.complex64)
    power = (
        spectrum.real.astype(np.float64) ** 2 + spectrum.imag.astype(np.float64) ** 2
    )
    energy = np.maximum(power @ spec.mel_filters, 1e-10)
    if spec.clap:
        return (10.0 * np.log10(energy)).astype(np.float32)
    log = np.log10(energy).T.astype(np.float32)
    return (
        np.maximum(log, log.max() - np.float32(8.0)) + np.float32(4.0)
    ) / np.float32(4.0)


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
