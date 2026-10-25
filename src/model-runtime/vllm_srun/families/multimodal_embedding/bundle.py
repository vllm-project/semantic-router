"""Vela Omni prepared bundles: the manifest, its file inventory and the pinned tensor contract.

``tools/models/vela_omni`` exports four ONNX graphs (text, image, CLAP, audio)
and the exact processor constants from a pinned source revision, and promotes
the export only after reference parity. A bundle is served only when its
manifest equals the variant's pinned contract, every file is listed with a
matching SHA-256 (nothing extra, no source code or source weights), and its
parity receipt passed every required modality case. Nothing in a bundle is
executed: the graphs run in ONNX Runtime without custom operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import inventory, read_json, sha256_file, sha256_json

MANIFEST = "vela_omni_manifest.json"
PENDING_MANIFEST = "vela_omni_manifest.pending.json"
GRAPHS = ("text", "image", "clap", "audio")
FORBIDDEN_SUFFIXES = (".py", ".pyc", ".safetensors", ".bin", ".pt", ".pth")
CLAP_DIMENSION = 512


@dataclass(frozen=True)
class Variant:
    """The pinned contract of one Omni size."""

    dimension: int
    max_tokens: int
    image_size: int
    resample: str
    text_pooling: str
    instruction_api: str | None
    sample_rates: tuple[int, ...]
    audio_cases: int


VARIANTS = {
    "nano": Variant(384, 512, 512, "bicubic", "cls", None, (16000, 44100, 48000), 4),
    "mini": Variant(
        768,
        32768,
        384,
        "bilinear",
        "last_token_full_l2_prefix_l2",
        "qwen_optional_instruction_v1",
        (),
        5,
    ),
}


@dataclass(frozen=True)
class OmniBundle:
    """A verified bundle. ``model_sha256`` identifies its exact file inventory."""

    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    model_sha256: str
    variant: str
    contract: Variant

    @property
    def source(self) -> tuple[str, str]:
        source = self.manifest["source"]
        return source["repo_id"], source["revision"]

    @property
    def graphs(self) -> dict[str, Path]:
        return {
            name: self.root / self.manifest["graphs"][name]["file"] for name in GRAPHS
        }

    @property
    def processors(self) -> dict[str, Any]:
        processors: dict[str, Any] = self.manifest["processors"]
        return processors

    def file(self, name: str) -> Path:
        return self.root / name


def is_bundle(root: Path) -> bool:
    return (Path(root) / MANIFEST).is_file()


def _port(name: str, dtype: str, shape: list[int | str]) -> dict[str, Any]:
    return {"name": name, "dtype": dtype, "shape": shape}


def expected_manifest(variant: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """The variant's pinned contract, completed with the bundle's own source, tokenizer facts and files."""
    spec = VARIANTS[variant]
    size, dimension = spec.image_size, spec.dimension
    inputs = {
        "text": [
            _port("input_ids", "int64", [1, "sequence"]),
            _port("attention_mask", "int64", [1, "sequence"]),
        ],
        "image": [_port("pixel_values", "float32", [1, 3, size, size])],
        "clap": [_port("input_features", "float32", [1, 1, 1001, 64])],
        "audio": [
            _port("input_features", "float32", [1, 80, 3000]),
            _port("clap_embedding", "float32", [1, CLAP_DIMENSION]),
        ],
    }
    text = manifest.get("processors", {}).get("text", {})
    return {
        "format_version": 1,
        "adapter": "vela_omni",
        "variant": variant,
        "source": manifest.get("source"),
        "embedding": {
            "dimension": dimension,
            "dimensions": [dimension],
            "normalization": "l2",
            "text_pooling": spec.text_pooling,
        },
        "tokenizer": "components/text/tokenizer.json",
        "max_text_length": spec.max_tokens,
        "graphs": {
            name: {
                "file": f"onnx/{name}.onnx",
                "inputs": ports,
                "output": _port(
                    "embedding",
                    "float32",
                    [1, CLAP_DIMENSION if name == "clap" else dimension],
                ),
            }
            for name, ports in inputs.items()
        },
        "processors": {
            "text": {
                "padding_side": text.get("padding_side"),
                "pad_token_id": text.get("pad_token_id"),
                "strip_whitespace": variant == "mini",
                "reject_overflow": True,
                "instruction_api": spec.instruction_api,
            },
            "image": {
                "size": size,
                "mean": [0.5] * 3,
                "std": [0.5] * 3,
                "resample": spec.resample,
            },
            "audio": {
                "file": "processors/audio.json",
                "max_seconds": 30,
                "sample_rates": list(spec.sample_rates),
                "max_sample_rate": 384000,
                "resample": {
                    "method": "sinc_interp_hann",
                    "lowpass_filter_width": 6,
                    "rolloff": 0.99,
                },
            },
        },
        "files": manifest.get("files"),
        "reference_parity": {"file": "reference_parity.json", "passed": True},
    }


def required_checks(variant: str) -> set[str]:
    checks = {
        "text/0",
        "text/1",
        "text/2",
        "text/3",
        "text/overflow-rejected",
        "text/padding",
    }
    checks |= {f"image/{index}" for index in range(3)}
    checks |= {
        f"audio/{index}/{part}"
        for index in range(VARIANTS[variant].audio_cases)
        for part in ("end-to-end", "clap")
    }
    if variant == "mini":
        checks |= {"text/instruction-query", "text/instruction-document"}
    return checks


def load(root: Path) -> OmniBundle:
    """Verify a prepared bundle; nothing in it is imported or executed."""
    root = Path(root).resolve()
    manifest_path = root / MANIFEST
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("variant") not in VARIANTS:
        raise PackageError("not a Vela Omni bundle manifest")
    variant = manifest["variant"]
    text = manifest.get("processors", {}).get("text", {})
    if text.get("padding_side") not in ("left", "right") or not (
        type(text.get("pad_token_id")) is int and text["pad_token_id"] >= 0
    ):
        raise PackageError("the bundle tokenizer needs a padding side and a pad token")
    if manifest != expected_manifest(variant, manifest):
        raise PackageError("the bundle differs from the pinned Omni tensor contract")
    source = manifest["source"]
    if not (isinstance(source, dict) and set(source) == {"repo_id", "revision"}):
        raise PackageError("the bundle does not name its source revision")
    if (root / PENDING_MANIFEST).exists():
        raise PackageError("the bundle still holds an unverified pending manifest")
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise PackageError("the bundle inventory is empty")
    for name in files:
        if PurePosixPath(name).suffix in FORBIDDEN_SUFFIXES:
            raise PackageError(
                f"the bundle holds source code or source weights: {name}"
            )
    actual = inventory(root)
    actual.pop(MANIFEST, None)
    if actual != files:
        extra = sorted(set(actual) - set(files))
        changed = sorted(name for name in files if actual.get(name) != files[name])
        raise PackageError(
            f"bundle files differ from its inventory (extra {extra[:3]}, changed {changed[:3]})"
        )
    required = {graph["file"] for graph in manifest["graphs"].values()} | {
        manifest["tokenizer"],
        manifest["processors"]["audio"]["file"],
        manifest["reference_parity"]["file"],
    }
    if not required <= files.keys():
        raise PackageError("the bundle inventory omits a graph or processor")
    report = read_json(root / manifest["reference_parity"]["file"])
    tests = report.get("tests") if isinstance(report, dict) else None
    if (
        not tests
        or report.get("passed") is not True
        or report.get("source") != source
        or report.get("variant") != variant
        or any(test.get("passed") is not True for test in tests)
        or not required_checks(variant) <= {test.get("name") for test in tests}
    ):
        raise PackageError(
            "the bundle's reference parity receipt is incomplete or failed"
        )
    return OmniBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_file(manifest_path),
        model_sha256=sha256_json(files),
        variant=variant,
        contract=VARIANTS[variant],
    )
