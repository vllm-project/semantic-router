"""Versioned, fully specified Vela Omni tensor artifact contract (no ML imports)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

WHISPER_SAMPLE_RATE = 16000
CLAP_SAMPLE_RATE = 48000
CLAP_WINDOW_SAMPLES = 10 * CLAP_SAMPLE_RATE
MAX_AUDIO_SAMPLES = 30 * CLAP_SAMPLE_RATE

FORMAT_VERSION = 1
MANIFEST = "vela_omni_manifest.json"
PENDING_MANIFEST = "vela_omni_manifest.pending.json"
VARIANTS = {
    "nano": {
        "dimension": 384,
        "max_tokens": 512,
        "image_size": 512,
        "resample": "bicubic",
    },
    "mini": {
        "dimension": 768,
        "max_tokens": 32768,
        "image_size": 384,
        "resample": "bilinear",
    },
}


def sources() -> dict:
    return json.loads(Path(__file__).with_name("sources.json").read_text())


def safe_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"unsafe artifact path: {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact escapes directory: {name!r}")
    return path


def digest(path: Path, algorithm: str = "sha256") -> str:
    if algorithm not in ("sha256", "git-blob-sha1"):
        raise ValueError(f"unsupported digest algorithm: {algorithm}")
    result = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-blob-sha1":
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def tensor(name: str, dtype: str, shape: list) -> dict:
    return {"name": name, "dtype": dtype, "shape": shape}


def graph_contracts(variant: str) -> dict:
    spec = VARIANTS[variant]
    dimension, size = spec["dimension"], spec["image_size"]
    inputs = {
        "text": [
            tensor("input_ids", "int64", [1, "sequence"]),
            tensor("attention_mask", "int64", [1, "sequence"]),
        ],
        "image": [tensor("pixel_values", "float32", [1, 3, size, size])],
        "clap": [tensor("input_features", "float32", [1, 1, 1001, 64])],
        "audio": [
            tensor("input_features", "float32", [1, 80, 3000]),
            tensor("clap_embedding", "float32", [1, 512]),
        ],
    }
    return {
        name: {
            "file": f"onnx/{name}.onnx",
            "inputs": values,
            "output": tensor(
                "embedding", "float32", [1, 512 if name == "clap" else dimension]
            ),
        }
        for name, values in inputs.items()
    }


def artifact_manifest(variant: str, padding_side: str, pad_token_id: int) -> dict:
    spec, source = VARIANTS[variant], sources()[variant]
    if (
        padding_side not in ("left", "right")
        or type(pad_token_id) is not int
        or pad_token_id < 0
    ):
        raise ValueError(
            "tokenizer must declare its padding side and nonnegative pad token"
        )
    return {
        "format_version": FORMAT_VERSION,
        "adapter": "vela_omni",
        "variant": variant,
        "source": {"repo_id": source["repo_id"], "revision": source["revision"]},
        "embedding": {
            "dimension": spec["dimension"],
            "dimensions": [spec["dimension"]],
            "normalization": "l2",
            "text_pooling": (
                "cls" if variant == "nano" else "last_token_full_l2_prefix_l2"
            ),
        },
        "tokenizer": "components/text/tokenizer.json",
        "max_text_length": spec["max_tokens"],
        "graphs": graph_contracts(variant),
        "processors": {
            "text": {
                "padding_side": padding_side,
                "pad_token_id": pad_token_id,
                "strip_whitespace": variant == "mini",
                "reject_overflow": True,
                "instruction_api": (
                    "qwen_optional_instruction_v1" if variant == "mini" else None
                ),
            },
            "image": {
                "size": spec["image_size"],
                "mean": [0.5] * 3,
                "std": [0.5] * 3,
                "resample": spec["resample"],
            },
            "audio": {
                "file": "processors/audio.json",
                "max_seconds": 30,
                "sample_rates": [16000, 44100, 48000] if variant == "nano" else [],
                "max_sample_rate": 384000,
                "resample": {
                    "method": "sinc_interp_hann",
                    "lowpass_filter_width": 6,
                    "rolloff": 0.99,
                },
            },
        },
        "files": {},
        "reference_parity": {"file": "reference_parity.json", "passed": False},
    }


def inventory(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in (MANIFEST, PENDING_MANIFEST)
    }


def verify_inventory(directory: Path, manifest: dict) -> None:
    if not manifest["files"]:
        raise ValueError("artifact inventory is empty")
    for name, expected in manifest["files"].items():
        if digest(safe_file(directory, name)) != expected:
            raise ValueError(f"artifact digest mismatch: {name}")


def endpoint_windows(samples: int) -> list[tuple[int, int]]:
    """The published 48 kHz endpoint-cover segmentation; never drops the tail."""
    if not 1 <= samples <= MAX_AUDIO_SAMPLES:
        raise ValueError("CLAP requires 1..1440000 samples at 48 kHz")
    if samples <= CLAP_WINDOW_SAMPLES:
        return [(0, samples)]
    count, last = (
        samples + CLAP_WINDOW_SAMPLES - 1
    ) // CLAP_WINDOW_SAMPLES, samples - CLAP_WINDOW_SAMPLES
    return [
        (i * last // (count - 1), i * last // (count - 1) + CLAP_WINDOW_SAMPLES)
        for i in range(count)
    ]
