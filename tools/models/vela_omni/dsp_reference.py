#!/usr/bin/env python3
"""Produce model-free DSP parity vectors from the exact published processors.

Only component JSON/tokenizer/source files are downloaded; no model weights.
This isolates audio resampling, mel feature extraction and image resizing from
the learned graphs. Full exported-model parity remains mandatory separately.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from contract import VARIANTS, WHISPER_SAMPLE_RATE, write_json
from PIL import Image
from processors import export_processors, prepare_audio, store_array, waveform_fixture
from source import provision
from transformers import (
    AutoImageProcessor,
    ClapFeatureExtractor,
    WhisperFeatureExtractor,
)


def probes(array) -> dict:
    flat = np.asarray(array, dtype=np.float32).flatten()
    indices = np.unique(
        np.linspace(0, len(flat) - 1, min(96, len(flat)), dtype=np.int64)
    )
    return {
        "shape": list(np.shape(array)),
        "indices": indices.tolist(),
        "values": flat[indices].tolist(),
    }


def jpeg_reference(processor, path: Path, output: Path, name: str) -> dict:
    source = Image.open(path)
    rgb = source.convert("RGB")
    pixels = processor(images=[rgb], return_tensors="np")["pixel_values"]
    return {
        "file": path.name,
        "source_mode": source.mode,
        "exif_orientation": source.getexif().get(274),
        "decoded_rgb": store_array(output, f"{name}-rgb", np.asarray(rgb)),
        "pixels": store_array(output, f"{name}-pixels", pixels),
        "probes": probes(pixels),
    }


def generate(variant: str, source: Path, output: Path, image_files=()):
    whisper = WhisperFeatureExtractor.from_pretrained(
        source / "components/audio", local_files_only=True
    )
    clap = ClapFeatureExtractor.from_pretrained(
        source / "components/audio_clap", local_files_only=True
    )
    image = AutoImageProcessor.from_pretrained(
        source / "components/image", local_files_only=True, use_fast=False
    )
    model = SimpleNamespace(
        audio_encoder=SimpleNamespace(feature_extractor=whisper),
        audio_processor=whisper,
        audio_residual=SimpleNamespace(processor=clap),
    )
    reference = SimpleNamespace(model=model, variant=variant)
    output.mkdir(parents=True, exist_ok=True)
    export_processors(reference, source, output)
    # Only reviewed source helpers are imported, authenticated by provision().
    sys.path.insert(0, str(source))
    records = []
    for index, rate in enumerate((16000, 44100, 48000)):
        waveform = waveform_fixture(rate, 0.04, stereo=rate != WHISPER_SAMPLE_RATE)
        audio16, audio48, windows, features, clap_features = prepare_audio(
            reference, waveform, rate
        )
        record = {
            "sampling_rate": rate,
            "windows": windows,
            "pcm": store_array(output, f"audio-{index}-pcm", waveform),
            "audio16": store_array(output, f"audio-{index}-16k", audio16),
            "audio48": store_array(output, f"audio-{index}-48k", audio48),
            "whisper_features": store_array(
                output, f"audio-{index}-whisper", features.numpy()
            ),
            "clap_features": store_array(
                output, f"audio-{index}-clap", clap_features.numpy()
            ),
            "probes": {
                "pcm": probes(waveform),
                "audio16": probes(audio16),
                "audio48": probes(audio48),
                "whisper": probes(features.numpy()),
                "clap": probes(clap_features.numpy()),
            },
        }
        records.append(record)
    yy, xx = np.indices((19, 23))
    rgb = np.stack(
        ((xx * 13 + yy * 7) % 256, ((xx // 3 + yy // 5) % 2) * 255, (xx * yy) % 256),
        axis=-1,
    ).astype(np.uint8)
    picture = Image.fromarray(rgb)
    picture.save(output / "image.png")
    pixels = image(images=[picture], return_tensors="np")["pixel_values"]
    picture.save(output / "image.jpg", quality=85, subsampling=2)
    variants = []
    for mode in ("L", "CMYK"):
        path = output / f"image-{mode.lower()}.jpg"
        picture.convert(mode).save(path, quality=85)
        variants.append(jpeg_reference(image, path, output, f"image-{mode.lower()}"))
    for index, source_image in enumerate(image_files):
        path = output / f"fixture-{index}.jpg"
        shutil.copyfile(source_image, path)
        variants.append(jpeg_reference(image, path, output, f"fixture-{index}"))
    result = {
        "format_version": 1,
        "variant": variant,
        "audio": records,
        "image": {
            "file": "image.png",
            "pixels": store_array(output, "image-pixels", pixels),
            "probes": probes(pixels),
        },
        "jpeg": jpeg_reference(image, output / "image.jpg", output, "image-jpeg"),
        "jpeg_variants": variants,
    }
    write_json(output / "index.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    source = provision(args.variant, args.source, args.download, weights=False)
    generate(args.variant, source, args.output, args.image_file)


if __name__ == "__main__":
    main()
