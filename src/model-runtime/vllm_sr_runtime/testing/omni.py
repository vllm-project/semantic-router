"""A tiny Vela Omni prepared bundle with the pinned tensor contract and toy graphs.

The manifest, inventory and parity receipt follow ``tools/models/vela_omni``
exactly; the four graphs are small projections with the real input and output
ports (``testing.onnx_graphs``), so verification, every processor and the
serving path run on a laptop CPU in a second.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..families.multimodal_embedding import bundle as bundles
from ..registry.artifacts import inventory
from . import onnx_graphs

SOURCE = {
    "repo_id": "vllm-sr/Vela-1.0-Omni-Nano",
    "revision": "2ff2d66385dbdd661a560ec3e8bcb45a0527d92e",
}
VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "route",
    "this",
    "request",
    "to",
    "a",
    "model",
    "hello",
]


def mel_filters(bins: int, mels: int) -> list[list[float]]:
    """Triangular filters over ``bins`` frequency bins: nonnegative, none empty."""
    edges = np.linspace(0, bins - 1, mels + 2)
    positions = np.arange(bins)[:, None]
    left, centre, right = edges[:-2], edges[1:-1], edges[2:]
    rising = (positions - left) / np.maximum(centre - left, 1e-9)
    falling = (right - positions) / np.maximum(right - centre, 1e-9)
    return np.clip(np.minimum(rising, falling), 0, None).round(6).tolist()


def audio_config() -> dict[str, Any]:
    common = {
        "window": "periodic_hann",
        "center": True,
        "pad_mode": "reflect",
        "power": 2,
        "floor": 1e-10,
    }
    return {
        "format_version": 1,
        "whisper": {
            **common,
            "sampling_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_samples": 480000,
            "n_frames": 3000,
            "mel_filters": mel_filters(201, 80),
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
            "mel_filters": mel_filters(513, 64),
            "log": "db",
            "reference": 1,
            "min_value": 1e-10,
            "db_range": None,
            "padding": "repeatpad",
        },
        "windows": "endpoint_cover_v1",
    }


def write_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))


def write_bundle(
    root: Path,
    *,
    variant: str = "nano",
    normalize: bool = True,
    source: dict[str, str] | None = None,
) -> Path:
    """Write a verified bundle of ``variant``'s contract at ``root``.

    ``source`` defaults to the variant's built-in pin; an unpinned source keeps
    the pin's recorded golden answers away from the random weights.
    """
    contract = bundles.VARIANTS[variant]
    size, dimension = contract.image_size, contract.dimension
    root.mkdir(parents=True, exist_ok=True)
    graphs = {
        "text": (
            {"input_ids": [1, "sequence"], "attention_mask": [1, "sequence"]},
            dimension,
        ),
        "image": ({"pixel_values": [1, 3, size, size]}, dimension),
        "clap": ({"input_features": [1, 1, 1001, 64]}, bundles.CLAP_DIMENSION),
        "audio": (
            {
                "input_features": [1, 80, 3000],
                "clap_embedding": [1, bundles.CLAP_DIMENSION],
            },
            dimension,
        ),
    }
    for seed, (name, (inputs, width)) in enumerate(graphs.items()):
        onnx_graphs.projection_graph(
            root / f"onnx/{name}.onnx", inputs, width, seed=seed, normalize=normalize
        )
    write_tokenizer(root / "components/text")
    (root / "processors").mkdir(exist_ok=True)
    (root / "processors/audio.json").write_text(
        json.dumps(audio_config()), encoding="utf-8"
    )
    if source is None:
        source = dict(SOURCE)
        if variant == "mini":
            source = {
                "repo_id": "vllm-sr/Vela-1.0-Omni-Mini",
                "revision": "801bae3ad28df6891408f0e0441c676b30e132e3",
            }
    checks = sorted(bundles.required_checks(variant))
    receipt = {
        "passed": True,
        "source": source,
        "variant": variant,
        "tests": [{"name": name, "passed": True} for name in checks],
    }
    (root / "reference_parity.json").write_text(json.dumps(receipt), encoding="utf-8")
    draft = {
        "source": source,
        "processors": {"text": {"padding_side": "right", "pad_token_id": 0}},
        "files": {},
    }
    manifest = bundles.expected_manifest(variant, draft)
    manifest["files"] = inventory(root)
    (root / bundles.MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return root
