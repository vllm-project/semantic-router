"""The Decision 3.0 package format (``d3-package-manifest/1``), verified without running package code.

A package is a Transformers ``Qwen3_5Model`` checkpoint (the text model and,
for image inputs, its vision tower), a 255-way answer-code readout
(``readout.safetensors``), ``decision_config.json`` (prompt family, answer
codes, attention mode, temperature, input limit), the tokenizer, chat template
and image processor settings, a card, and bundled Python that this runtime
never imports. ``MODEL_MANIFEST.json`` lists the SHA-256 and size of every
file, the parameter counts and the identity of the scored checkpoint. The
checks reproduce the released runtime's ``verify_package`` and the package
builder's ``model_identity``; only the files the family loads are read.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import (
    named_files,
    read_json,
    safetensors_header,
    sha256_file,
)

MANIFEST_NAME = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "d3-package-manifest/1"
FORMAT_ID = "d3-code-readout-v1"
FORMAT_VERSION = 1
PROMPTS = ("d3",)
ATTENTION_MODES = ("causal", "noncausal_full_attention")
READOUT_DTYPES = ("float32", "bfloat16")
MAX_OPTIONS = 255
MIN_INPUT_TOKENS = 16
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# decision_config.json fields that change answers; the identity covers them.
INFERENCE_FIELDS = (
    "format_version",
    "format_id",
    "prompt",
    "base_model",
    "revision",
    "codes",
    "token_ids",
    "temperature",
    "attention_mode",
    "pooling",
    "max_length",
    "readout_dtype",
)
# Files besides the weights that the identity covers.
IDENTITY_FILES = (
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
# Every file the family reads, besides the weights.
MODEL_FILES = (
    "config.json",
    "decision_config.json",
    *IDENTITY_FILES,
)
IMAGE_FILES = ("preprocessor_config.json",)
WEIGHT_INDEX = "model.safetensors.index.json"
VISION_PREFIXES = ("visual.", "model.visual.")
TEXT_PREFIXES = ("language_model.", "model.language_model.")


@dataclass(frozen=True)
class Decision3Package:
    """What the family needs after verification."""

    root: Path
    manifest: dict[str, Any]
    decision_config: dict[str, Any]
    config: dict[str, Any]
    weights: tuple[str, ...]
    text_prefix: str
    vision_prefix: str | None
    processor_config: dict[str, Any] | None
    model_sha256: str
    max_input_tokens: int


def is_package(root: Path) -> bool:
    """A Decision 3.0 export: ``decision_config.json`` names the code-readout format and the d3 prompt."""
    path = Path(root) / "decision_config.json"
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return (
        isinstance(config, dict)
        and config.get("format_id") == FORMAT_ID
        and config.get("prompt") in PROMPTS
    )


def weight_files(root: Path) -> tuple[str, ...]:
    """The checkpoint's weight files as the package builder lists them (the index first, then its shards)."""
    index = root / WEIGHT_INDEX
    if index.is_file():
        mapping = read_json(index, mapping=True).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise PackageError(f"{WEIGHT_INDEX} has no weight_map")
        return (WEIGHT_INDEX, *sorted(set(mapping.values())))
    if (root / "model.safetensors").is_file():
        return ("model.safetensors",)
    raise PackageError("the package has no model*.safetensors weights")


def model_identity(files: dict[str, str], decision_config: dict[str, Any]) -> str:
    """``model_identity`` of the package builder: the weight and identity file hashes plus the inference fields."""
    fields = {
        key: decision_config[key] for key in INFERENCE_FIELDS if key in decision_config
    }
    blob = json.dumps(
        {"files": files, "decision": fields}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def read_manifest(root: Path) -> tuple[dict[str, Any], str]:
    manifest = read_json(root / MANIFEST_NAME, mapping=True)
    files = manifest.get("files_sha256")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or not isinstance(files, dict)
        or not files
        or MANIFEST_NAME in files
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in files.values()
        )
    ):
        raise PackageError("unknown or malformed Decision 3.0 package manifest")
    sizes = manifest.get("files_bytes")
    if sizes is not None and not isinstance(sizes, dict):
        raise PackageError("manifest files_bytes must be an object")
    return manifest, sha256_file(root / MANIFEST_NAME)


def verify_files(
    root: Path, manifest: dict[str, Any], names: list[str]
) -> dict[str, str]:
    """SHA-256 of every file the family loads; each must match ``MODEL_MANIFEST.json`` in size and digest."""
    listed: dict[str, str] = manifest["files_sha256"]
    sizes: dict[str, Any] = manifest.get("files_bytes") or {}
    unlisted = sorted(name for name in names if name not in listed)
    if unlisted:
        raise PackageError(f"MODEL_MANIFEST.json does not list {unlisted[:5]}")
    for name in names:
        path = root / name
        if name in sizes and path.is_file() and path.stat().st_size != sizes[name]:
            raise PackageError(f"{name}: size differs from MODEL_MANIFEST.json")
    actual = named_files(root, names)
    changed = sorted(name for name in names if actual[name] != listed[name])
    if changed:
        raise PackageError(
            f"package files differ from MODEL_MANIFEST.json: {changed[:5]}"
        )
    return actual


def check_decision_config(config: Any) -> dict[str, Any]:
    """The released runtime's checks of ``decision_config.json``; returns it."""
    if not isinstance(config, dict):
        raise PackageError("decision_config.json must be an object")
    if (
        config.get("format_version") != FORMAT_VERSION
        or config.get("format_id") != FORMAT_ID
    ):
        raise PackageError("unsupported decision_config.json format")
    if config.get("prompt", "d3") not in PROMPTS:
        raise PackageError(f"unknown prompt family {config.get('prompt')!r}")
    if config.get("attention_mode", "causal") not in ATTENTION_MODES:
        raise PackageError(f"unknown attention mode {config.get('attention_mode')!r}")
    if config.get("pooling", "last") != "last":
        raise PackageError(f"unsupported pooling {config.get('pooling')!r}")
    if config.get("readout_dtype", "float32") not in READOUT_DTYPES:
        raise PackageError("readout_dtype must be float32 or bfloat16")
    temperature = config.get("temperature", 1.0)
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise PackageError("temperature must be positive and finite")
    codes, token_ids = config.get("codes"), config.get("token_ids")
    if (
        not isinstance(codes, list)
        or not isinstance(token_ids, list)
        or len(codes) != MAX_OPTIONS
        or len(token_ids) != MAX_OPTIONS
        or any(not isinstance(code, str) or not code for code in codes)
        or any(type(token) is not int or token < 0 for token in token_ids)
        or len(set(codes)) != MAX_OPTIONS
        or len(set(token_ids)) != MAX_OPTIONS
    ):
        raise PackageError(
            "decision_config.json needs 255 distinct answer codes and token IDs"
        )
    limit = config.get("max_length")
    if limit is not None and (type(limit) is not int or limit < MIN_INPUT_TOKENS):
        raise PackageError(
            f"max_length must be null or an integer >= {MIN_INPUT_TOKENS}"
        )
    return config


def check_config(config: Any) -> dict[str, Any]:
    """The checkpoint's ``config.json``: a Qwen3.5 model with a text config (and a vision config)."""
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        raise PackageError("config.json must describe a qwen3_5 model")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_text":
        raise PackageError("config.json needs a qwen3_5_text text_config")
    vision = config.get("vision_config")
    if vision is not None and (
        not isinstance(vision, dict) or vision.get("model_type") != "qwen3_5_vision"
    ):
        raise PackageError("config.json vision_config must be a qwen3_5_vision config")
    return config


def tensor_groups(
    root: Path, weights: tuple[str, ...]
) -> tuple[dict[str, int], str, str | None]:
    """Parameter counts per group (text, vision) from the safetensors headers, and the two name prefixes."""
    counts: dict[str, int] = {}
    prefixes: dict[str, set[str]] = {"text": set(), "vision": set()}
    for name in weights:
        if not name.endswith(".safetensors"):
            continue
        for tensor, meta in safetensors_header(root / name).items():
            if tensor == "__metadata__":
                continue
            shape = meta.get("shape") if isinstance(meta, dict) else None
            if not isinstance(shape, list) or any(
                type(dim) is not int or dim < 0 for dim in shape
            ):
                raise PackageError(f"invalid tensor shape in {name}: {tensor}")
            vision = next((p for p in VISION_PREFIXES if tensor.startswith(p)), None)
            text = next((p for p in TEXT_PREFIXES if tensor.startswith(p)), None)
            if vision is None and text is None:
                raise PackageError(
                    f"checkpoint tensor {tensor!r} belongs to no known module"
                )
            group = "vision" if vision else "text"
            prefixes[group].add(vision or text or "")
            counts[group] = counts.get(group, 0) + math.prod(shape)
    if len(prefixes["text"]) != 1 or len(prefixes["vision"]) > 1:
        raise PackageError("checkpoint tensors use mixed name prefixes")
    vision_prefix = next(iter(prefixes["vision"]), None)
    return counts, next(iter(prefixes["text"])), vision_prefix


def readout_shape(root: Path) -> list[int]:
    header = {
        name: meta
        for name, meta in safetensors_header(root / "readout.safetensors").items()
        if name != "__metadata__"
    }
    if set(header) != {"weight"} or not isinstance(header["weight"], dict):
        raise PackageError(
            "readout.safetensors must hold exactly one tensor named weight"
        )
    shape = header["weight"].get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise PackageError("the readout weight must be a matrix")
    return shape


def check_processor(config: Any) -> dict[str, Any]:
    """``preprocessor_config.json`` of the Qwen2-VL image processor with the settings the family reproduces."""
    if not isinstance(config, dict):
        raise PackageError("preprocessor_config.json must be an object")
    if config.get("image_processor_type") not in (
        "Qwen2VLImageProcessorFast",
        "Qwen2VLImageProcessor",
    ):
        raise PackageError(
            f"unsupported image processor {config.get('image_processor_type')!r}"
        )
    defaults = {
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "do_convert_rgb": True,
        "resample": 3,
        "rescale_factor": 1 / 255,
    }
    changed = sorted(
        key for key, value in defaults.items() if config.get(key, value) != value
    )
    if changed:
        raise PackageError(f"unsupported image processor settings: {changed}")
    for key in ("patch_size", "temporal_patch_size", "merge_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise PackageError(f"preprocessor_config.json needs a positive {key}")
    for key in ("image_mean", "image_std"):
        values = config.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise PackageError(f"preprocessor_config.json needs three {key} values")
    if any(v == 0 for v in config["image_std"]):
        raise PackageError("image_std must not contain zero")
    return config


def verify(root: Path) -> tuple[Decision3Package, str]:
    """Verify a package; returns its details and the manifest's SHA-256."""
    manifest, manifest_sha256 = read_manifest(root)
    decision = check_decision_config(read_json(root / "decision_config.json"))
    config = check_config(read_json(root / "config.json"))
    weights = weight_files(root)
    vision_config = config.get("vision_config")
    names = [*MODEL_FILES, *weights]
    if vision_config is not None and (root / IMAGE_FILES[0]).is_file():
        names.append(IMAGE_FILES[0])
    hashes = verify_files(root, manifest, names)
    identity_files = {
        name: hashes[name]
        for name in (
            *[w for w in weights if w.endswith(".safetensors")],
            *[n for n in IDENTITY_FILES if (root / n).is_file()],
        )
    }
    model_sha256 = model_identity(identity_files, decision)
    declared = manifest.get("identity")
    if not isinstance(declared, dict) or declared.get("model_sha256") != model_sha256:
        raise PackageError(
            "model identity differs from the scored checkpoint in MODEL_MANIFEST.json"
        )
    counts, text_prefix, vision_prefix = tensor_groups(root, weights)
    hidden = config["text_config"].get("hidden_size")
    shape = readout_shape(root)
    if shape != [MAX_OPTIONS, hidden]:
        raise PackageError(
            f"readout weight has shape {shape}, expected {[MAX_OPTIONS, hidden]}"
        )
    parameters = manifest.get("parameters")
    readout = MAX_OPTIONS * hidden
    expected = {
        "text": counts.get("text", 0),
        "vision": counts.get("vision", 0),
        "readout": readout,
        "loaded": counts.get("text", 0) + counts.get("vision", 0) + readout,
    }
    if not isinstance(parameters, dict) or any(
        parameters.get(key) != value for key, value in expected.items()
    ):
        raise PackageError("tensor counts differ from the manifest's parameter counts")
    processor = None
    if vision_prefix is not None:
        if vision_config is None:
            raise PackageError("the checkpoint has vision weights but no vision_config")
        if IMAGE_FILES[0] in names:
            processor = check_processor(read_json(root / IMAGE_FILES[0]))
    limit = decision.get("max_length") or config["text_config"].get(
        "max_position_embeddings"
    )
    if type(limit) is not int or limit < MIN_INPUT_TOKENS:
        raise PackageError("the package declares no usable input limit")
    details = Decision3Package(
        root=root,
        manifest=manifest,
        decision_config=decision,
        config=config,
        weights=weights,
        text_prefix=text_prefix,
        vision_prefix=vision_prefix,
        processor_config=processor,
        model_sha256=model_sha256,
        max_input_tokens=limit,
    )
    return details, manifest_sha256
