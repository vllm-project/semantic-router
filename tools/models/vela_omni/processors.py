# Phase-local imports preserve memory limits and authenticated source loading.
# ruff: noqa: PLC0415
"""Export exact processor constants and produce native preprocessing references."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from contract import (
    CLAP_SAMPLE_RATE,
    WHISPER_SAMPLE_RATE,
    endpoint_windows,
    sources,
    write_json,
)
from torch.nn import functional


def speech_processor(reference):
    return (
        reference.model.audio_encoder.feature_extractor
        if reference.variant == "nano"
        else reference.model.audio_processor
    )


def image_processor(reference):
    return (
        reference.model.image_encoder.processor
        if reference.variant == "nano"
        else reference.model.image_processor
    )


def export_processors(reference, source: Path, output: Path) -> None:
    whisper = speech_processor(reference)
    clap = reference.model.audio_residual.processor
    common = {
        "window": "periodic_hann",
        "center": True,
        "pad_mode": "reflect",
        "power": 2,
        "floor": 1e-10,
    }
    value = {
        "format_version": 1,
        "whisper": {
            **common,
            "sampling_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_samples": 480000,
            "n_frames": 3000,
            "mel_filters": whisper.mel_filters.tolist(),
            "log": "log10",
            "range": 8,
            "affine": [0.25, 1.0],
        },
        "clap": {
            **common,
            "sampling_rate": 48000,
            "n_fft": 1024,
            "hop_length": 480,
            "n_samples": 480000,
            "n_frames": 1001,
            "mel_filters": clap.mel_filters_slaney.tolist(),
            "log": "db",
            "reference": 1,
            "min_value": 1e-10,
            "db_range": None,
            "padding": "repeatpad",
        },
        "windows": "endpoint_cover_v1",
    }
    if (
        whisper.sampling_rate != WHISPER_SAMPLE_RATE
        or clap.sampling_rate != CLAP_SAMPLE_RATE
        or clap.padding != "repeatpad"
        or clap.truncation != "rand_trunc"
    ):
        raise ValueError("published audio preprocessing changed")
    write_json(output / "processors/audio.json", value)
    for name in sources()[reference.variant]["files"]:
        if name.startswith("components/text/") and not name.endswith(".safetensors"):
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
    shutil.copyfile(source / "config.json", output / "source_config.json")


def prepare_audio(reference, waveform: np.ndarray, sampling_rate: int):
    # Use the authenticated official helper as the export-time reference only.
    if reference.variant == "nano":
        from omni_components.tiny_clap_residual import (
            native_rate,
        )
    else:
        from omni_components.medium_clap_residual import (
            native_rate,
        )
    audio16 = native_rate(waveform, sampling_rate, 16000)
    audio48 = native_rate(waveform, sampling_rate, 48000)
    whisper = speech_processor(reference)(
        [audio16], sampling_rate=16000, return_tensors="pt"
    )["input_features"]
    windows = endpoint_windows(len(audio48))
    chunks = [audio48[start:end] for start, end in windows]
    clap = reference.model.audio_residual.processor(
        chunks, sampling_rate=48000, return_tensors="pt"
    )
    if clap["is_longer"].any() or tuple(clap["input_features"].shape[1:]) != (
        1,
        1001,
        64,
    ):
        raise ValueError(
            "CLAP endpoint windows did not produce the declared unfused inputs"
        )
    return audio16, audio48, windows, whisper.float(), clap["input_features"].float()


def aggregate_clap(vectors: torch.Tensor) -> torch.Tensor:
    # Single-window readout is already normalized; preserve the public order.
    return (
        vectors[:1]
        if len(vectors) == 1
        else functional.normalize(vectors.mean(0, keepdim=True), dim=-1)
    )


def waveform_fixture(rate: int, seconds: float, stereo: bool = False) -> np.ndarray:
    """Deterministic PCM with distinct head/middle/tail and high-frequency energy."""
    count = round(rate * seconds)
    t = np.arange(count, dtype=np.float64) / rate
    wave = 0.2 * np.sin(2 * np.pi * 317 * t) + 0.1 * np.sin(
        2 * np.pi * min(11000, rate * 0.4) * t
    )
    wave += (t > seconds * 0.6) * 0.15 * np.sin(2 * np.pi * 1793 * t)
    wave *= 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2
    if stereo:
        wave = np.stack((wave, wave * 0.7 + 0.07 * np.sin(2 * np.pi * 811 * t)))
    return wave.astype(np.float32)


def store_array(root: Path, name: str, values) -> dict:
    array = np.asarray(values, dtype="<f4")
    target = root / (name + ".f32")
    target.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(target)
    return {
        "file": target.relative_to(root).as_posix(),
        "dtype": "float32_le",
        "shape": list(array.shape),
    }
