"""Fail-closed JevArena release packager for qualified native models.

This is a byte-integrity and process gate, not a GPU parity runner. The input
model directory must already run using its embedded native inference code. An
external-base PEFT profile retains the scored adapter and decision head without
copying or merging its pinned upstream backbone.
Gold and raw training rows remain outside the public package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import adapter_runtime
from .adapter_parity import MAX_DRIFT as ADAPTER_MAX_DRIFT
from .adapter_parity import VERSION as ADAPTER_PARITY_VERSION
from .generate_arena import ARTIFACTS, matched_models
from .generate_arena import VERSION as ARTIFACT_VERSION

VERSION = "decision2-jevarena-release-bundle/2"
EXTERNAL_ADAPTER = "qwen-external-base-peft"
RECORD_VERSION = "decision2-release-package-record/2"
PARITY_VERSION = "decision2-native-package-parity/1"
GATE_VERSION = "decision2-jevarena-release-gate/1"
AUTHORED_EDITORIAL_VERSION = "decision2-authored-editorial-receipt/1"
CANDIDATE_FREEZE_VERSION = "decision2-candidate-freeze-audit/1"
PRETEST_FREEZE_VERSION = "decision2-jevarena-pretest-freeze/1"
MODEL_ID = re.compile(r"llm-semantic-router/DEV2\.0-(?:0\.6B|0\.8B|2B|4B|9B|27B)\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
IMMUTABLE_REVISION = re.compile(r"[a-f0-9]{40}(?:[a-f0-9]{24})?\Z")
HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SECRET = re.compile(
    r"(?i)(?:\bhf_[A-Za-z0-9]{20,}|\bjv_live_[A-Za-z0-9_-]{16,}"
    r"|\bapikey_[A-Za-z0-9_-]{16,}|\bsk-[A-Za-z0-9_-]{16,}"
    r"|Authorization\s*:\s*Bearer\s+\S+)"
)
PRIVATE_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:home|root|data|work|mnt|tmp|private|Users|var|opt)/[^\s\"'<>]+"
    r"|\b[A-Za-z]:[\\/](?:Users|Documents|ProgramData|Windows)[\\/][^\s\"'<>]+"
)
IP_ADDRESS = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
TEXT_SUFFIXES = {".json", ".py", ".md", ".txt", ".jinja", ".yaml", ".yml"}
MODEL_SUFFIXES = TEXT_SUFFIXES | {".safetensors", ".model", ".tiktoken"}
SPECIAL_TEXT = {"LICENSE", "LICENSE-Qwen", "NOTICE", "SHA256SUMS"}
QWEN_RUNTIME = {
    f"decision2/{name}.py"
    for name in (
        "__init__",
        "api",
        "calibration",
        "data",
        "decision_model",
        "infer",
        "lora",
        "source",
    )
}
SCORE_FAMILIES = ("synthetic", "css", "public", "dbv4", "authored")
GATE_CHECKS = (
    "candidate_freeze",
    "authored_editorial",
    "train_eval_overlap",
    "same_panel_evaluation",
    "rights_and_provenance",
    "native_parity",
    "release_thresholds",
)
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _object(path: Path) -> dict[str, Any]:
    return _json_snapshot(path)[0]


def _json_snapshot(path: Path) -> tuple[dict[str, Any], bytes, str]:
    """Parse and hash one read, so a later copy can use the validated bytes."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSON file: {path.name}")
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value, payload, hashlib.sha256(payload).hexdigest()


def _public_text(text: str, label: str) -> None:
    if SECRET.search(text) or PRIVATE_PATH.search(text) or IP_ADDRESS.search(text):
        raise ValueError(
            f"{label} contains private infrastructure or credential-like text"
        )


def _portable_name(name: str) -> bool:
    path = Path(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and all(part and not part.startswith(".") for part in path.parts)
        and str(path) == name
    )


def _inventory(root: Path, *, allow_adapter_metadata: bool = False) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Model input must be a regular directory")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if not _portable_name(name) or path.is_symlink():
            raise ValueError("Model input has a symlink or nonportable path")
        if path.is_dir():
            continue
        if not path.is_file() or (
            path.suffix not in MODEL_SUFFIXES and path.name not in SPECIAL_TEXT
        ):
            raise ValueError(f"Unsupported model package file: {name}")
        if path.name in {
            "README.md",
            "MODEL_MANIFEST.json",
            "publication-manifest.json",
        } and not (
            allow_adapter_metadata and name in {"README.md", "MODEL_MANIFEST.json"}
        ):
            raise ValueError("Model input must contain functional files only")
        if path.suffix in TEXT_SUFFIXES or path.name in SPECIAL_TEXT:
            _public_text(path.read_text(encoding="utf-8"), name)
        files[name] = sha_file(path)
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Model input has no safe weights")
    return files


def _tensor_counts(path: Path) -> dict[str, int]:
    """Inspect only a safetensors header; require consistent shapes and offsets."""
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        length = int.from_bytes(raw, "little")
        if not 2 <= length <= 128 << 20:
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        header = stream.read(length)
        if len(header) != length:
            raise ValueError(f"Truncated safetensors header: {path.name}")
    data = json.loads(header)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid safetensors header: {path.name}")
    _public_text(json.dumps(data, ensure_ascii=False), path.name)
    payload = path.stat().st_size - 8 - length
    intervals: list[tuple[int, int]] = []
    counts: dict[str, int] = {}
    for name, tensor in data.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(name, str) or not name or not isinstance(tensor, dict):
            raise ValueError("Invalid safetensors tensor entry")
        shape, offsets, dtype = (
            tensor.get("shape"),
            tensor.get("data_offsets"),
            tensor.get("dtype"),
        )
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or dtype not in DTYPE_BYTES
        ):
            raise ValueError(f"Invalid safetensors tensor shape or dtype: {name}")
        count = math.prod(shape)
        start, end = offsets
        if end - start != count * DTYPE_BYTES[dtype] or end > payload:
            raise ValueError(f"Safetensors offsets disagree with shape: {name}")
        intervals.append((start, end))
        counts[name] = count
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"Safetensors payload has a gap or overlap: {path.name}")
        cursor = end
    if cursor != payload or not counts:
        raise ValueError(f"Safetensors payload is incomplete: {path.name}")
    return counts


def _profile(root: Path, architecture: str, files: dict[str, str]) -> None:
    names = set(files)
    if architecture == EXTERNAL_ADAPTER:
        required = {
            "MODEL_MANIFEST.json",
            "README.md",
            "requirements.txt",
            "calibration.json",
            "model/decision_config.json",
            "model/decision_head.safetensors",
            "model/adapter/adapter_config.json",
            "model/adapter/adapter_model.safetensors",
            "model/tokenizer.json",
        } | QWEN_RUNTIME
        if any(name.startswith("model/backbone/") for name in names):
            raise ValueError("External-base PEFT package may not copy backbone weights")
    elif architecture in {"qwen3.5-decision-head", "qwen3.8-decision-head"}:
        required = {
            "model/decision_config.json",
            "model/materialization_receipt.json",
            "model/decision_head.safetensors",
            "model/backbone/config.json",
            "model/tokenizer.json",
            "calibration.json",
            "requirements.txt",
            "LICENSE",
        } | QWEN_RUNTIME
        weight_prefix = "model/backbone/"
        if not any(
            name.startswith(weight_prefix) and name.endswith(".safetensors")
            for name in names
        ):
            raise ValueError("Qwen decision-head package lacks merged backbone weights")
        config = _object(root / "model/decision_config.json")
        if config.get("checkpoint_format") != "full":
            raise ValueError(
                "Qwen decision-head package must contain full merged weights"
            )
    elif architecture == "qwen3.5-semif":
        required = {
            "serve.py",
            "decision_config.json",
            "config.json",
            "calib.json",
            "tokenizer.json",
            "SHA256SUMS",
            "decision2_provenance.json",
            "LICENSE",
            "LICENSE-Qwen",
            "NOTICE",
        }
        if not any("/" not in name and name.endswith(".safetensors") for name in names):
            raise ValueError("Native SemIf package lacks merged model weights")
    elif architecture == "encoder-decision":
        required = {
            "model.safetensors",
            "encoder/config.json",
            "rl_agent_config.json",
            "LICENSE",
        }
        if not ({"serve.py", "runtime.py"} & names):
            raise ValueError("Encoder package lacks a native inference entrypoint")
        if not any(name.startswith("tokenizer/") for name in names):
            raise ValueError("Encoder package lacks a tokenizer")
        if "CHECKPOINT.json" in names:
            checkpoint = _object(root / "CHECKPOINT.json")
            if (
                checkpoint.get("research_only") is not False
                or checkpoint.get("release_qualified") is not True
            ):
                raise ValueError("Encoder research checkpoint is not release-qualified")
    else:
        raise ValueError("Unrecognized release architecture")
    if not required <= names:
        raise ValueError(f"{architecture} package lacks {sorted(required - names)}")
    if architecture not in {"encoder-decision", EXTERNAL_ADAPTER} and any(
        "adapter_model.safetensors" in name for name in names
    ):
        raise ValueError("A partial LoRA adapter is not a full publishable model")


def _parameter_count(
    root: Path,
    active: list[str],
    support: list[str],
    excluded: list[str],
    files: dict[str, str],
) -> int:
    all_weights = {name for name in files if name.endswith(".safetensors")}
    if (
        not isinstance(active, list)
        or not active
        or not isinstance(support, list)
        or not isinstance(excluded, list)
        or any(not isinstance(name, str) for name in (*active, *support, *excluded))
        or len(set(active + support)) != len(active + support)
        or set(active + support) != all_weights
    ):
        raise ValueError(
            "Every safe weight file must be classified as active or support"
        )
    tensor_counts: dict[str, int] = {}
    for name in active:
        for tensor, count in _tensor_counts(root / name).items():
            key = f"{name}::{tensor}"
            if key in tensor_counts:
                raise ValueError("Duplicate active tensor identity")
            tensor_counts[key] = count
    for name in support:
        _tensor_counts(root / name)
    if len(set(excluded)) != len(excluded) or not set(excluded) <= set(tensor_counts):
        raise ValueError("Excluded buffers must name unique active tensors")
    count = sum(value for key, value in tensor_counts.items() if key not in excluded)
    if count < 1:
        raise ValueError("No active model parameters")
    return count


def _size_compatible(count: int, model_id: str) -> bool:
    nominal = float(model_id.rsplit("-", 1)[-1][:-1])
    return abs(count / 1e9 - nominal) / nominal <= 0.25


def _native_identity(
    root: Path,
    record: dict[str, Any],
    files: dict[str, str],
    external_manifest: dict[str, Any] | None = None,
) -> str:
    identity = record.get("native_identity")
    if not isinstance(identity, dict) or set(identity) != {"scheme", "sha256", "file"}:
        raise ValueError("Native model identity declaration is incomplete")
    expected = _sha(identity["sha256"], "native model identity")
    scheme, name = identity["scheme"], identity["file"]
    if scheme == "external-peft-checkpoint-fingerprint":
        if name is not None or external_manifest is None:
            raise ValueError("External PEFT identity needs a verified adapter manifest")
        actual = external_manifest["model_sha256"]
    elif scheme == "qwen-checkpoint-fingerprint":
        if name is not None:
            raise ValueError("Qwen checkpoint fingerprint has no identity file")
        from training.model.infer import checkpoint_fingerprint

        actual = checkpoint_fingerprint(root / "model")["model_sha256"]
    elif scheme == "sha256-file":
        if not isinstance(name, str) or not _portable_name(name) or name not in files:
            raise ValueError("Native identity file is missing")
        if name == "SHA256SUMS":
            listed = {}
            for line in (root / name).read_text(encoding="utf-8").splitlines():
                digest, separator, relative = line.partition("  ")
                if (
                    separator != "  "
                    or not _portable_name(relative)
                    or relative in listed
                    or not SHA256.fullmatch(digest)
                ):
                    raise ValueError("Malformed native SHA256SUMS")
                listed[relative] = digest
            if listed != {key: value for key, value in files.items() if key != name}:
                raise ValueError(
                    "Native SHA256SUMS does not cover exact functional files"
                )
        actual = files[name]
    else:
        raise ValueError("Unknown native identity scheme")
    if actual != expected:
        raise ValueError("Native model identity differs from package bytes")
    return actual


def _external_adapter_contract(
    root: Path,
    base_source: Path,
    record: dict[str, Any],
    files: dict[str, str],
) -> tuple[dict[str, Any], int, dict[str, str]]:
    """Bind external bytes and count the loaded base, LoRA and native head."""
    if base_source.is_symlink():
        raise ValueError("External base source cannot be a directory symlink")
    # The source-aware verifier imports .infer from the *copied* decision2
    # package. Calling its source module here would resolve publication.infer.
    from .adapter_bundle import _verify_staged_runtime

    manifest = adapter_runtime.verify_bundle(root)
    _verify_staged_runtime(root, base_source)
    from training.model.infer import checkpoint_fingerprint

    identity = checkpoint_fingerprint(root / "model", base_source)
    if identity["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("External PEFT scored checkpoint identity changed")
    if files != {
        **manifest["files_sha256"],
        "MODEL_MANIFEST.json": files["MODEL_MANIFEST.json"],
    }:
        raise ValueError("External PEFT files differ from frozen native package")
    base = manifest["base"]
    if (
        manifest.get("model_id") != record["model_id"]
        or base["repo_id"] != record["base_model"]["id"]
        or base["revision"] != record["base_model"]["revision"]
    ):
        raise ValueError("External PEFT base or model ID differs from release record")
    active = record.get("active_weight_files")
    if (
        set(active or [])
        != {
            "model/adapter/adapter_model.safetensors",
            "model/decision_head.safetensors",
        }
        or record.get("support_weight_files") != []
        or record.get("non_parameter_tensors") != []
    ):
        raise ValueError("External PEFT weights must be only the adapter and head")
    local_count = _parameter_count(root, active, [], [], files)
    breakdown = manifest.get("parameter_breakdown")
    if (
        not isinstance(breakdown, dict)
        or local_count != breakdown.get("adapter", -1) + breakdown.get("head", -1)
        or breakdown.get("base_text", 0) < 1
        or manifest.get("parameter_count") != local_count + breakdown["base_text"]
        or breakdown.get("total") != manifest["parameter_count"]
    ):
        raise ValueError("External PEFT parameter inventory differs from loaded model")
    return manifest, manifest["parameter_count"], identity["files_sha256"]


def _adapter_source_parity(
    path: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
) -> str:
    """Require the separate scored-source versus unmerged package BF16 check."""
    receipt = _object(path)
    expected = {
        "package_manifest_sha256": manifest_sha,
        "scored_prediction_manifest_sha256": manifest[
            "scored_prediction_manifest_sha256"
        ],
        "scored_predictions_sha256": manifest["scored_predictions_sha256"],
        "model_sha256": manifest["model_sha256"],
        "calibration_sha256": manifest["calibration_sha256"],
    }
    if (
        receipt.get("schema_version") != ADAPTER_PARITY_VERSION
        or receipt.get("passed") is not True
        or any(receipt.get(name) != value for name, value in expected.items())
        or receipt.get("types") != ["choice", "noul", "score"]
        or type(receipt.get("items")) is not int
        or receipt["items"] < 1
        or type(receipt.get("questions")) is not int
        or receipt["questions"] < 3
        or receipt.get("invalid_or_missing_n") != 0
        or receipt.get("categorical_mismatch_n") != 0
        or receipt.get("predeclared_gate")
        != {
            "invalid_or_missing_n": 0,
            "categorical_mismatch_n": 0,
            "max_probability_or_score_drift": ADAPTER_MAX_DRIFT,
        }
    ):
        raise ValueError("External PEFT source parity is missing or unbound")
    drift = receipt.get("max_probability_or_score_drift")
    if (
        type(drift) not in (int, float)
        or not math.isfinite(drift)
        or not 0 <= drift <= ADAPTER_MAX_DRIFT
    ):
        raise ValueError("External PEFT source parity drift exceeds the fixed gate")
    runtime = receipt.get("native_runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("bf16") is not True
        or runtime.get("one_item_batch") is not True
        or runtime.get("no_truncation") is not True
    ):
        raise ValueError("External PEFT source parity used another runtime")
    for name in (
        "gold_free_prompts_sha256",
        "source_answers_sha256",
        "package_answers_sha256",
    ):
        _sha(receipt.get(name), name)
    _public_text(path.read_text(encoding="utf-8"), "adapter source parity receipt")
    return sha_file(path)


def _qwen_runtime_contract(root: Path, native_sha: str) -> dict[str, Any]:
    """Produce the existing portable runtime's inner verifier manifest."""
    from training.model.calibration import (
        load_calibration,
        verified_materialization_origin,
    )
    from training.model.infer import checkpoint_fingerprint

    checkpoint = root / "model"
    model = checkpoint_fingerprint(checkpoint)
    if model["model_sha256"] != native_sha:
        raise ValueError("Qwen runtime checkpoint differs from evaluated weights")
    origin = verified_materialization_origin(checkpoint, native_sha)
    if origin is None:
        raise ValueError("Qwen runtime lacks exact merged-checkpoint lineage")
    temperatures, report = load_calibration(
        root / "calibration.json",
        native_sha,
        materialized_source_sha256=origin["source_model_sha256"],
    )
    max_length = report.get("inference", {}).get("max_length")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Qwen runtime CAL lacks an audited context limit")
    return {
        "bundle_version": "decision2-self-contained-bundle/1",
        "model_sha256": native_sha,
        "model_files_sha256": model["files_sha256"],
        "source_model_sha256": origin["source_model_sha256"],
        "materialization_sha256": origin["receipt_sha256"],
        "calibration_sha256": sha_file(root / "calibration.json"),
        "temperature_by_type": temperatures,
        "max_length": max_length,
    }


def _verify_qwen_runtime(root: Path) -> None:
    """Import only the copied package's verifier; no model weights are loaded."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "decision2",
        root / "decision2" / "__init__.py",
        submodule_search_locations=[str(root / "decision2")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Copied Decision 2.0 runtime cannot be imported")
    previous = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "decision2" or name.startswith("decision2.")
    }
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules["decision2"] = module
        spec.loader.exec_module(module)
        module.verify_bundle(root)
    finally:
        sys.dont_write_bytecode = previous_bytecode
        for name in list(sys.modules):
            if name == "decision2" or name.startswith("decision2."):
                del sys.modules[name]
        sys.modules.update(previous)


def _rights(record: dict[str, Any], model_id: str, revision: str) -> None:
    if record.get("schema_version") != RECORD_VERSION:
        raise ValueError("Unknown release provenance record")
    if record.get("model_id") != model_id or record.get("model_revision") != revision:
        raise ValueError("Training provenance names another model")
    source = record.get("base_model", {})
    if (
        not isinstance(source, dict)
        or not HF_ID.fullmatch(str(source.get("id", "")))
        or not IMMUTABLE_REVISION.fullmatch(str(source.get("revision", "")))
        or not isinstance(source.get("license"), str)
        or not source["license"]
    ):
        raise ValueError("Base model needs an immutable revision and reviewed license")
    training = record.get("training", {})
    if not isinstance(training, dict) or any(
        type(training.get(field)) is not int or training[field] <= 0
        for field in ("train_rows", "select_rows", "cal_rows")
    ):
        raise ValueError("Training partition counts are missing")
    languages = training.get("language_counts")
    if (
        not isinstance(languages, dict)
        or not languages
        or any(
            not isinstance(language, str)
            or not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", language)
            or type(count) is not int
            or count <= 0
            for language, count in languages.items()
        )
        or sum(languages.values()) != training["train_rows"]
    ):
        raise ValueError("TRAIN language counts must sum to train_rows")
    for field in (
        "data_manifest_sha256",
        "run_provenance_sha256",
        "training_code_sha256",
    ):
        _sha(training.get(field), field)
    if (
        not isinstance(training.get("selection_policy"), str)
        or not training["selection_policy"]
    ):
        raise ValueError("Training selection policy is missing")
    rights = record.get("rights", {})
    if (
        not isinstance(rights, dict)
        or rights.get("status") != "passed"
        or rights.get("no_raw_rows") is not True
        or rights.get("scope")
        not in {"noncommercial_research_weights_card", "unrestricted_weights_card"}
        or not isinstance(rights.get("reviewed_by"), str)
        or not rights["reviewed_by"].strip()
        or not isinstance(rights.get("sources"), list)
        or not rights["sources"]
    ):
        raise ValueError("Reviewed source rights and release scope are required")
    for source in rights["sources"]:
        if not isinstance(source, dict) or any(
            not isinstance(source.get(field), str) or not source[field].strip()
            for field in (
                "name",
                "license",
                "attribution",
                "use_scope",
                "redistribution",
            )
        ):
            raise ValueError("Every training source needs rights and attribution")
    if (
        rights["scope"] == "noncommercial_research_weights_card"
        and record.get("license_id") != "other"
    ):
        raise ValueError(
            "Restricted source terms cannot be advertised as an open license"
        )
    if not isinstance(record.get("limitations"), list) or not record["limitations"]:
        raise ValueError("The model card must disclose limitations")
    if (
        not isinstance(record.get("evaluation_language_scope"), str)
        or not record["evaluation_language_scope"].strip()
    ):
        raise ValueError("Evaluation language coverage must be disclosed")
    if not isinstance(record.get("known_overlap"), list):
        raise ValueError("Known train/evaluation overlap needs an explicit disclosure")
    if (
        record.get("license_id") not in {"other", "apache-2.0", "mit", "cc-by-4.0"}
        or not isinstance(record.get("native_adapter_version"), str)
        or not record["native_adapter_version"].strip()
        or any(
            not isinstance(value, str) or not value.strip()
            for value in (*record["limitations"], *record["known_overlap"])
        )
    ):
        raise ValueError("Release license, adapter or disclosures are invalid")


def _release_artifacts(
    artifacts: Path,
    arena_rank: Path,
    public_rank: Path,
    model_id: str,
    revision: str,
    score_key: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if (
        manifest.get("publication_version") != ARTIFACT_VERSION
        or manifest.get("phase") != "release"
    ):
        raise ValueError("Publication requires completed six-axis release artifacts")
    hashes = manifest.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACTS):
        raise ValueError("Release artifact file inventory is incomplete")
    for name in ARTIFACTS:
        path = artifacts / name
        if (
            path.is_symlink()
            or not path.is_file()
            or sha_file(path) != _sha(hashes[name], name)
        ):
            raise ValueError(
                f"Release artifact differs from generator manifest: {name}"
            )
        _public_text(path.read_text(encoding="utf-8"), name)
    ranks = manifest.get("ranking_sha256", {})
    if (
        not isinstance(ranks, dict)
        or ranks.get("arena") != sha_file(arena_rank)
        or ranks.get("jevbench_public") != sha_file(public_rank)
    ):
        raise ValueError("Rank reports differ from release artifact generator inputs")
    arena, public = _object(arena_rank), _object(public_rank)
    if (
        arena.get("schema_version") != "jevarena-ranking/2"
        or arena.get("phase") != "release"
    ):
        raise ValueError("JevArena ranking is not a release ranking")
    if (
        public.get("schema_version") != "jevarena-jevbench-public-rank/1"
        or public.get("items") != 231
    ):
        raise ValueError("JevBench public ranking is not the pinned public panel")
    matched_models(arena, public)
    model_entries = manifest.get("models")
    if not isinstance(model_entries, list) or len(model_entries) != len(
        arena["models"]
    ):
        raise ValueError("Generator manifest has incomplete model coverage")
    by_key = {entry["key"]: entry for entry in model_entries}
    if len(by_key) != len(model_entries) or set(by_key) != {
        entry["key"] for entry in arena["models"]
    }:
        raise ValueError("Generator manifest has duplicate or missing models")
    by_public = {entry["key"]: entry for entry in public["models"]}
    for arena_entry in arena["models"]:
        emitted = by_key[arena_entry["key"]]
        if any(
            emitted.get(key) != arena_entry.get(key)
            for key in ("model_id", "revision", "size_b")
        ):
            raise ValueError("Generator manifest model identity differs from ranking")
        if emitted.get("arena_rank") != arena_entry.get("rank") or emitted.get(
            "jevbench_public_rank"
        ) != by_public[arena_entry["key"]].get("rank"):
            raise ValueError("Generator manifest ranks differ from scorer reports")
    selected = [row for row in arena.get("models", []) if row.get("key") == score_key]
    peer = [row for row in public.get("models", []) if row.get("key") == score_key]
    entry = [row for row in manifest.get("models", []) if row.get("key") == score_key]
    if len(selected) != 1 or len(peer) != 1 or len(entry) != 1:
        raise ValueError("Selected model is absent or duplicated in matched rankings")
    row = selected[0]
    for candidate in (row, peer[0], entry[0]):
        if (
            candidate.get("model_id") != model_id
            or candidate.get("revision") != revision
            or candidate.get("size_b") != row.get("size_b")
        ):
            raise ValueError("Artifact model identity or parameter count differs")
    if row.get("group") != "decision2" or peer[0].get("group") != "decision2":
        raise ValueError("Selected release row is not Decision 2.0")
    if set(row.get("axes", {})) != {
        "typed",
        "transfer",
        "jevbench_public",
        "decision_bench_v4",
        "sealed_authored",
        "robustness",
    } or row.get("report_sha256", {}).get("public") != peer[0].get("report_sha256"):
        raise ValueError("Selected model lacks a matched six-axis score")
    if manifest.get("panel_sha256") != arena.get("panel_sha256"):
        raise ValueError("Artifact panel fingerprint differs from ranking")
    if manifest.get("coverage") != row.get("coverage"):
        raise ValueError("Artifact coverage differs from selected release row")
    roster = sorted((entry["model_id"], entry["revision"]) for entry in arena["models"])
    if len(set(roster)) != len(roster):
        raise ValueError("JevArena candidate roster repeats a model revision")
    roster_sha = hashlib.sha256(
        json.dumps(roster, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest, row, roster_sha


def _score_inputs(
    paths: dict[str, dict[str, Path]],
    row: dict[str, Any],
    native_sha: str,
    model_id: str,
    revision: str,
    calibration_sha: str,
    adapter_version: str,
    package_manifest_sha: str | None = None,
    checkpoint_files: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    if not isinstance(paths, dict) or set(paths) != set(SCORE_FAMILIES):
        raise ValueError(
            "All five release score and native prediction receipts are required"
        )
    binding = {}
    for family in SCORE_FAMILIES:
        item = paths[family]
        if not isinstance(item, dict) or set(item) != {
            "score",
            "predictions",
            "native_manifest",
        }:
            raise ValueError(f"{family}: incomplete score binding")
        score_path, predictions, native_path = (
            item[key] for key in ("score", "predictions", "native_manifest")
        )
        if any(
            path.is_symlink() or not path.is_file()
            for path in (score_path, predictions, native_path)
        ):
            raise ValueError(f"{family}: scorer input must be a regular file")
        score, native = _object(score_path), _object(native_path)
        score_sha, prediction_sha, native_sha_file = (
            sha_file(score_path),
            sha_file(predictions),
            sha_file(native_path),
        )
        if score_sha != row.get("report_sha256", {}).get(family):
            raise ValueError(f"{family}: scorer report differs from JevArena ranking")
        if (
            score.get("predictions_sha256") != prediction_sha
            or native.get("predictions_sha256") != prediction_sha
        ):
            raise ValueError(f"{family}: predictions differ from scored native run")
        if (
            family in {"public", "dbv4", "authored"}
            and score.get("prediction_manifest_sha256") != native_sha_file
        ):
            raise ValueError(f"{family}: scorer used another native manifest")
        if family in {"synthetic", "css"} and score.get(
            "prediction_manifest_sha256"
        ) not in (None, native_sha_file):
            raise ValueError(f"{family}: scorer used another native manifest")
        native_calibration = native.get("calibration_sha256") or native.get(
            "calibration", {}
        ).get("file_sha256")
        if (
            native.get("model_id") != model_id
            or native.get("model_revision") != revision
            or native.get("model_sha256") != native_sha
            or native.get("adapter_version") != adapter_version
            or native_calibration != calibration_sha
        ):
            raise ValueError(f"{family}: native run used another model package")
        adapter_sha = _sha(native.get("adapter_sha256"), f"{family} native adapter")
        if (
            package_manifest_sha is not None
            and native.get("package_manifest_sha256") != package_manifest_sha
        ):
            raise ValueError(f"{family}: native run used another external PEFT package")
        if (
            checkpoint_files is not None
            and native.get("model_files_sha256") != checkpoint_files
        ):
            raise ValueError(f"{family}: native run used another external PEFT base")
        binding[family] = {
            "score_sha256": score_sha,
            "predictions_sha256": prediction_sha,
            "native_manifest_sha256": native_sha_file,
            "adapter_sha256": adapter_sha,
        }
        if family == "authored":
            binding[family]["selection_lock_sha256"] = _sha(
                score.get("selection_lock_sha256"), "authored selection lock"
            )
    return binding


def _parity(
    receipt: dict[str, Any],
    model_id: str,
    revision: str,
    native_sha: str,
    model_files_digest: str,
    calibration_sha: str,
) -> None:
    if (
        receipt.get("schema_version") != PARITY_VERSION
        or receipt.get("status") != "passed"
        or receipt.get("model_id") != model_id
        or receipt.get("model_revision") != revision
        or receipt.get("native_model_sha256") != native_sha
        or receipt.get("model_files_sha256") != model_files_digest
        or receipt.get("calibration_sha256") != calibration_sha
    ):
        raise ValueError("Native parity receipt does not bind released package")
    panels = receipt.get("panels", {})
    if not isinstance(panels, dict) or set(panels) != {"dev", "css_pilot"}:
        raise ValueError(
            "Native parity needs independent DEV and transfer pilot panels"
        )
    for name, count in (("dev", 1600), ("css_pilot", 1430)):
        panel = panels[name]
        if (
            not isinstance(panel, dict)
            or panel.get("items") != count
            or panel.get("answers") != count
            or panel.get("categorical_mismatch_n") != 0
            or panel.get("gate_pass") is not True
        ):
            raise ValueError(f"{name}: native parity is missing or failed")
        _sha(panel.get("prompt_sha256"), f"{name} prompts")
        _sha(panel.get("report_sha256"), f"{name} parity report")
        for field, limit in (
            ("probability_drift_p99", 0.005),
            ("probability_drift_max", 0.02),
        ):
            value = panel.get(field)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or value > limit
            ):
                raise ValueError(f"{name}: native probability parity threshold failed")


def _gate(
    gate: dict[str, Any],
    *,
    record_sha: str,
    parity_sha: str,
    artifact_sha: str,
    native_sha: str,
    model_files_digest: str,
    model_id: str,
    revision: str,
) -> None:
    expected = {
        "package_record_sha256": record_sha,
        "parity_receipt_sha256": parity_sha,
        "artifact_manifest_sha256": artifact_sha,
        "native_model_sha256": native_sha,
        "model_files_sha256": model_files_digest,
    }
    if (
        gate.get("schema_version") != GATE_VERSION
        or gate.get("status") != "passed"
        or gate.get("model_id") != model_id
        or gate.get("model_revision") != revision
        or any(gate.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Release gate is missing, blocked, or bound to other bytes")
    _sha(gate.get("pretest_freeze_sha256"), "pretest freeze")
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(GATE_CHECKS):
        raise ValueError("Release gate omits a required review")
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("status") != "passed":
            raise ValueError(f"Release gate review is blocked: {name}")
        _sha(check.get("evidence_sha256"), f"{name} evidence")


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} needs an explicit UTC time")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} needs an explicit UTC time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"{label} needs an explicit UTC time")
    return parsed


def _review_row(
    review: Any,
    allowed: list[Any],
    packet_at: datetime,
    key_at: datetime,
) -> tuple[str, Any]:
    fields = {
        "reviewer_identity_sha256",
        "review_sha256",
        "sealed_at_utc",
        "native_answer",
        "source_a_evidence",
        "source_b_evidence",
        "both_sources_necessary",
        "ambiguity",
        "document_realism",
        "shortcut_risk",
        "rights_concern",
        "all_paragraphs_checked",
        "paragraph_notes",
    }
    if not isinstance(review, dict) or set(review) != fields:
        raise ValueError("Authored review is not a complete row-level judgment")
    reviewer = _sha(review["reviewer_identity_sha256"], "reviewer identity")
    _sha(review["review_sha256"], "sealed review")
    sealed = _utc(review["sealed_at_utc"], "review seal")
    if not packet_at < sealed < key_at:
        raise ValueError("Authored review was not sealed before key access")
    answer = review["native_answer"]
    if not any(type(answer) is type(option) and answer == option for option in allowed):
        raise ValueError("Authored reviewer gave an invalid native answer")
    if any(
        not isinstance(review[field], str) or not review[field].strip()
        for field in ("source_a_evidence", "source_b_evidence", "paragraph_notes")
    ) or (
        review["both_sources_necessary"] is not True
        or review["ambiguity"] != "none"
        or review["document_realism"] != "plausible"
        or review["shortcut_risk"] != "none"
        or review["rights_concern"] is not False
        or review["all_paragraphs_checked"] is not True
    ):
        raise ValueError("Authored review has unresolved editorial concerns")
    return reviewer, answer


def _allowed_answers(kind: str, allowed: Any) -> list[Any]:
    if not isinstance(allowed, list) or not 2 <= len(allowed) <= 64:
        raise ValueError("Authored native answer vocabulary is invalid")
    if kind == "choice":
        valid = all(isinstance(value, str) and value.strip() for value in allowed)
        valid = valid and len(set(allowed)) == len(allowed)
    elif kind == "noul":
        valid = allowed in ([False, True], [True, False])
    else:
        valid = all(type(value) is int for value in allowed) and allowed == list(
            range(len(allowed))
        )
    if not valid:
        raise ValueError("Authored native answer vocabulary is invalid")
    return allowed


def _authored_editorial(
    evidence: dict[str, Any],
    freeze: dict[str, Any],
    freeze_sha: str,
    panel: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    """Check receipt completeness and consistency, not the reviewers' humanity."""
    count = coverage.get("sealed_authored_items")
    if (
        evidence.get("schema_version") != AUTHORED_EDITORIAL_VERSION
        or evidence.get("status") != "passed"
        or evidence.get("pretest_freeze_sha256") != freeze_sha
        or evidence.get("authored_prompts_sha256") != panel.get("authored_prompts")
        or evidence.get("authored_targets_sha256") != panel.get("authored_targets")
        or freeze.get("status") != "frozen"
        or freeze.get("authored_prompts_sha256") != panel.get("authored_prompts")
        or freeze.get("authored_targets_sha256") != panel.get("authored_targets")
        or evidence.get("original_roster_sha256")
        != freeze.get("original_roster_sha256")
        or evidence.get("authored_native_contracts_sha256")
        != freeze.get("authored_native_contracts_sha256")
        or type(count) is not int
        or not 1200 <= count <= 1480
        or evidence.get("independent_originals") != count
    ):
        raise ValueError("Authored editorial receipt is absent or not bound to freeze")
    packet_at = _utc(evidence.get("packet_sealed_at_utc"), "blind packet seal")
    key_at = _utc(evidence.get("key_opened_at_utc"), "editorial key access")
    completed_at = _utc(evidence.get("completed_at_utc"), "adjudication completion")
    if not packet_at < key_at < completed_at:
        raise ValueError("Authored editorial chronology is invalid")
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Authored editorial lacks every independent row")
    ids: set[str] = set()
    native_contracts: list[list[Any]] = []
    kinds: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    forms: set[str] = set()
    templates: Counter[str] = Counter()
    bands: Counter[str] = Counter()
    second_by_kind: Counter[str] = Counter()
    fields = {
        "original_id_sha256",
        "source_family_sha256",
        "author_identity_sha256",
        "type",
        "original_allowed_answers",
        "paired_allowed_answers",
        "domain",
        "operation",
        "form_family",
        "template_sha256",
        "length_band",
        "original_review_a",
        "original_review_b",
        "paired_review",
        "paired_second_review",
        "adjudication",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Authored editorial row is incomplete")
        original = _sha(row["original_id_sha256"], "authored original")
        _sha(row["source_family_sha256"], "source family")
        author = _sha(row["author_identity_sha256"], "authored author")
        template = _sha(row["template_sha256"], "document template")
        if original in ids:
            raise ValueError("Authored editorial repeats an original")
        ids.add(original)
        kind = row["type"]
        if not isinstance(kind, str) or kind not in {"choice", "noul", "score"}:
            raise ValueError("Authored editorial has an unknown native type")
        original_allowed = _allowed_answers(kind, row["original_allowed_answers"])
        paired_allowed = _allowed_answers(kind, row["paired_allowed_answers"])
        native_contracts.append([original, kind, original_allowed, paired_allowed])
        for name in ("domain", "operation", "form_family"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError("Authored editorial omits allocation provenance")
        if row["length_band"] not in {"short", "medium", "long"}:
            raise ValueError("Authored editorial omits a valid length band")
        kinds[kind] += 1
        domains[row["domain"]] += 1
        operations[row["operation"]] += 1
        forms.add(row["form_family"])
        templates[template] += 1
        bands[row["length_band"]] += 1
        reviews = [
            _review_row(row[name], original_allowed, packet_at, key_at)
            for name in ("original_review_a", "original_review_b")
        ]
        reviews.append(
            _review_row(row["paired_review"], paired_allowed, packet_at, key_at)
        )
        if row["paired_second_review"] is not None:
            reviews.append(
                _review_row(
                    row["paired_second_review"], paired_allowed, packet_at, key_at
                )
            )
            second_by_kind[kind] += 1
        adjudication = row["adjudication"]
        if not isinstance(adjudication, dict) or set(adjudication) != {
            "adjudicator_identity_sha256",
            "adjudication_sha256",
            "completed_at_utc",
            "verdict",
            "original_answer",
            "paired_answer",
            "oracle_agreement",
            "semantic_independence_passed",
            "overlap_passed",
            "provenance_passed",
            "rights_passed",
            "unresolved_material_errors",
        }:
            raise ValueError("Authored row has no complete adjudication")
        adjudicator = _sha(
            adjudication["adjudicator_identity_sha256"], "adjudicator identity"
        )
        _sha(adjudication["adjudication_sha256"], "adjudication receipt")
        adjudicated_at = _utc(adjudication["completed_at_utc"], "adjudication")
        if not key_at < adjudicated_at <= completed_at:
            raise ValueError("Authored adjudication predates key access")
        if (
            len({author, adjudicator, *(identity for identity, _ in reviews)})
            != len(reviews) + 2
        ):
            raise ValueError("Authored author, reviewers and adjudicator overlap")
        if (
            any(answer != adjudication["original_answer"] for _, answer in reviews[:2])
            or any(answer != adjudication["paired_answer"] for _, answer in reviews[2:])
            or adjudication["verdict"] != "accepted"
            or any(
                adjudication[name] is not True
                for name in (
                    "oracle_agreement",
                    "semantic_independence_passed",
                    "overlap_passed",
                    "provenance_passed",
                    "rights_passed",
                )
            )
            or type(adjudication["unresolved_material_errors"]) is not int
            or adjudication["unresolved_material_errors"] != 0
        ):
            raise ValueError("Authored row has unresolved adjudication")
    roster_sha = hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":")).encode()
    ).hexdigest()
    if roster_sha != _sha(evidence.get("original_roster_sha256"), "authored roster"):
        raise ValueError("Authored editorial row IDs differ from frozen roster")
    native_contracts_sha = hashlib.sha256(
        json.dumps(
            sorted(native_contracts, key=lambda item: item[0]),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if native_contracts_sha != _sha(
        evidence.get("authored_native_contracts_sha256"), "authored native contracts"
    ):
        raise ValueError("Authored native answer contracts differ from freeze")
    if (
        evidence.get("type_counts") != dict(kinds)
        or any(kinds[kind] < 360 for kind in ("choice", "noul", "score"))
        or len(domains) < 12
        or len(forms) < 9
        or any(value > count * 0.12 for value in domains.values())
        or any(value > count * 0.05 for value in operations.values())
        or any(value > count * 0.03 for value in templates.values())
        or not 0.25 <= bands["short"] / count <= 0.35
        or not 0.30 <= bands["medium"] / count <= 0.45
        or not 0.25 <= bands["long"] / count <= 0.40
        or any(second_by_kind[kind] < math.ceil(kinds[kind] * 0.15) for kind in kinds)
    ):
        raise ValueError(
            "Authored editorial allocation or second-review coverage failed"
        )


def _candidate_freeze(
    freeze: dict[str, Any],
    freeze_sha: str,
    evidence: dict[str, Any],
    candidate: dict[str, Any],
    roster_sha: str,
    score_binding: dict[str, dict[str, str]],
) -> None:
    """Check declared freeze order and bytes; timestamp authenticity is external."""
    formula_sha = sha_file(
        Path(__file__).resolve().parents[1] / "jev_arena/arena_v2.py"
    )
    authored_lock = score_binding["authored"]["selection_lock_sha256"]
    if (
        freeze.get("schema_version") != PRETEST_FREEZE_VERSION
        or freeze.get("status") != "frozen"
        or freeze.get("candidate_roster_sha256") != roster_sha
        or freeze.get("selected_candidate") != candidate
        or freeze.get("formula_version") != "jevarena-ranking/2"
        or freeze.get("formula_sha256") != formula_sha
        or freeze.get("selection_lock_sha256") != authored_lock
        or evidence.get("schema_version") != CANDIDATE_FREEZE_VERSION
        or evidence.get("status") != "passed"
        or evidence.get("pretest_freeze_sha256") != freeze_sha
        or evidence.get("selected_candidate") != candidate
        or evidence.get("formula_sha256") != formula_sha
        or evidence.get("selection_lock_sha256") != authored_lock
    ):
        raise ValueError("Candidate freeze or formula does not bind evaluated model")
    frozen_at = _utc(freeze.get("frozen_at_utc"), "candidate freeze")
    label_opened_at = _utc(
        evidence.get("first_protected_label_opened_at_utc"), "formal label access"
    )
    _sha(evidence.get("timestamp_log_sha256"), "independent timestamp log")
    _sha(evidence.get("reviewer_identity_sha256"), "freeze reviewer identity")
    seals = evidence.get("protected_prediction_seals")
    if not isinstance(seals, dict) or set(seals) != {
        "synthetic",
        "css",
        "authored",
    }:
        raise ValueError("Protected FINAL predictions lack sealed receipts")
    for family, seal in seals.items():
        binding = score_binding[family]
        if (
            not isinstance(seal, dict)
            or set(seal)
            != {"native_manifest_sha256", "predictions_sha256", "sealed_at_utc"}
            or seal["native_manifest_sha256"] != binding["native_manifest_sha256"]
            or seal["predictions_sha256"] != binding["predictions_sha256"]
        ):
            raise ValueError(f"{family}: protected prediction seal differs")
        sealed_at = _utc(seal["sealed_at_utc"], f"{family} prediction seal")
        if not frozen_at < sealed_at < label_opened_at:
            raise ValueError(
                f"{family}: freeze, prediction and label chronology is invalid"
            )


def _external_evidence(
    record: dict[str, Any],
    gate: dict[str, Any],
    provenance_inputs: dict[str, Path],
    freeze_manifest: Path,
    gate_evidence: dict[str, Path],
    *,
    freeze: dict[str, Any],
    freeze_sha: str,
    panel: dict[str, Any],
    coverage: dict[str, Any],
    candidate: dict[str, Any],
    roster_sha: str,
    score_binding: dict[str, dict[str, str]],
) -> dict[str, str]:
    training = record["training"]
    expected = {
        "data_manifest": training["data_manifest_sha256"],
        "run_provenance": training["run_provenance_sha256"],
        "training_code": training["training_code_sha256"],
    }
    if not isinstance(provenance_inputs, dict) or set(provenance_inputs) != set(
        expected
    ):
        raise ValueError("Exact training provenance inputs are required")
    if not isinstance(gate_evidence, dict) or set(gate_evidence) != set(GATE_CHECKS):
        raise ValueError("Every release check requires its original evidence file")
    paths = {
        **provenance_inputs,
        "pretest_freeze": freeze_manifest,
        **{f"gate:{name}": path for name, path in gate_evidence.items()},
    }
    observed: dict[str, str] = {}
    authored_payload: bytes | None = None
    candidate_payload: bytes | None = None
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"External evidence is missing or linked: {name}")
        wanted = (
            expected[name]
            if name in expected
            else (
                gate["pretest_freeze_sha256"]
                if name == "pretest_freeze"
                else gate["checks"][name.removeprefix("gate:")]["evidence_sha256"]
            )
        )
        if name == "pretest_freeze":
            digest = freeze_sha
        else:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if name == "gate:authored_editorial":
                authored_payload = payload
            elif name == "gate:candidate_freeze":
                candidate_payload = payload
        if digest != wanted:
            raise ValueError(f"External evidence changed after review: {name}")
        observed[name] = digest
    assert authored_payload is not None
    authored = json.loads(authored_payload)
    if not isinstance(authored, dict):
        raise ValueError("Authored editorial receipt must be a JSON object")
    _authored_editorial(authored, freeze, freeze_sha, panel, coverage)
    assert candidate_payload is not None
    candidate_evidence = json.loads(candidate_payload)
    if not isinstance(candidate_evidence, dict):
        raise ValueError("Candidate freeze receipt must be a JSON object")
    _candidate_freeze(
        freeze, freeze_sha, candidate_evidence, candidate, roster_sha, score_binding
    )
    return observed


def _md(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("<", "&lt;").replace(">", "&gt;")
    for mark in "|[]`*_~#!(){}":
        escaped = escaped.replace(mark, "\\" + mark)
    return escaped.replace("\n", " ").replace("\r", " ")


def _card(
    model_id: str,
    record: dict[str, Any],
    row: dict[str, Any],
    count: int,
    score_table: str,
) -> str:
    rights = record["rights"]
    sources = "\n".join(
        "| "
        + " | ".join(
            _md(source[field])
            for field in (
                "name",
                "license",
                "attribution",
                "use_scope",
                "redistribution",
            )
        )
        + " |"
        for source in rights["sources"]
    )
    notes = "\n".join(f"- {_md(value)}" for value in record["limitations"])
    train_languages = "\n".join(
        f"| {_md(language)} | {count:,} | {count / record['training']['train_rows']:.1%} |"
        for language, count in sorted(record["training"]["language_counts"].items())
    )
    overlap = (
        "\n".join(f"- {_md(value)}" for value in record["known_overlap"])
        or "- No overlap declared in the reviewed record."
    )
    external_note = ""
    if record["architecture"] == EXTERNAL_ADAPTER:
        external_note = (
            "This repository contains the native PEFT adapter, tokenizer, "
            "calibration and Decision 2.0 head. The upstream base weights are "
            "an external dependency at the immutable revision above; they "
            "are not copied or merged into this repository. A generic "
            "`AutoModel` call will not run the custom decision head. Use "
            '`decision2.Decision2.from_pretrained("native", '
            'source_path="verified_base_snapshot")` after installing '
            "`native/requirements.txt`, or let the native loader retrieve "
            "and hash-check the pinned commit. The reported parameter count "
            "includes the loaded base text backbone, adapter and head.\n"
        )
    return f"""---
license: {record["license_id"]}
{("license_name: noncommercial-research-terms" + chr(10)) if record["license_id"] == "other" else ""}base_model: {record["base_model"]["id"]}
tags:
- decision-model
- typed-decision
- jevarena
---

![Decision 2.0 chibi mosaic owl with a three-facet decision gem](decision-2-sticker-chibi-v4.png)

# {model_id}

Native architecture: `{record["architecture"]}`. Actual model parameters:
**{count:,}**. Source model: [{record["base_model"]["id"]}](https://huggingface.co/{record["base_model"]["id"]})
at immutable revision `{record["base_model"]["revision"]}`. Native runtime and
calibration are included in this package; its exact invocation and limits are
described by the bundled runtime and `PACKAGE_MANIFEST.json`.

{external_note}

## Same-panel release evaluation

JevArena rank **#{row["rank"]}**, six-axis score **{row["score"]:.2f}** on the
frozen release panel. Missing and invalid answers count as misses. Ranks apply
only to the matched roster in the table. The public JevBench subset is an
independent 231-item rerun, not an official closed-set rank.

![JevArena ranking](jevarena-rank.svg)
![JevArena parameter Pareto plot](jevarena-pareto.svg)
![JevArena axis matrix](jevarena-axis-matrix.svg)
![JevArena model by task matrix](jevarena-task-matrix.svg)
![Public JevBench ranking](jevbench-public-rank.svg)
![Public JevBench parameter Pareto plot](jevbench-public-pareto.svg)

{score_table.rstrip()}

The six figures, frozen panel digests and paired intervals are bound in
`card-artifacts/manifest.json`.
Calibration, invalidity, speed and cost are reported separately when measured
under comparable conditions; this package does not invent missing measurements.

## Training, rights and limitations

TRAIN {record["training"]["train_rows"]:,}; SELECT {record["training"]["select_rows"]:,};
CAL {record["training"]["cal_rows"]:,}. Selection policy:
{_md(record["training"]["selection_policy"])}. Scope: `{rights["scope"]}`.
No raw upstream text or benchmark labels are bundled.

| TRAIN language | Rows | Share |
| --- | ---: | ---: |
{train_languages}

Evaluation language coverage: {_md(record["evaluation_language_scope"])}.

| Source | Terms | Attribution | Use | Redistribution |
| --- | --- | --- | --- | --- |
{sources}

Known training/evaluation overlap:
{overlap}

Limitations:
{notes}

`PACKAGE_MANIFEST.json` binds this card, copied weights, source/rights
record, gold-free native parity receipt and all five scored native runs.
The packager checks bytes and the declared release gate. It does not itself
rerun GPU inference or grant new rights in source data.
"""


def assemble(
    *,
    model_dir: Path,
    artifacts: Path,
    arena_rank: Path,
    public_rank: Path,
    package_record: Path,
    parity_receipt: Path,
    release_gate: Path,
    provenance_inputs: dict[str, Path],
    freeze_manifest: Path,
    gate_evidence: dict[str, Path],
    score_inputs: dict[str, dict[str, Path]],
    score_key: str,
    output: Path,
    base_source: Path | None = None,
    adapter_source_parity_receipt: Path | None = None,
) -> dict[str, Any]:
    """Validate a qualified native package and atomically stage public bytes."""
    for path in (
        model_dir,
        artifacts,
        arena_rank,
        public_rank,
        package_record,
        parity_receipt,
        release_gate,
    ):
        if path.is_symlink():
            raise ValueError("Input symlinks are not allowed")
    model_dir, artifacts = (
        model_dir.resolve(strict=True),
        artifacts.resolve(strict=True),
    )
    output = output.resolve()
    if (
        output.exists()
        or output.is_relative_to(model_dir)
        or output.is_relative_to(artifacts)
    ):
        raise FileExistsError("Output exists or is inside an input directory")
    record, record_bytes, record_sha = _json_snapshot(package_record)
    parity, parity_bytes, parity_sha = _json_snapshot(parity_receipt)
    gate, gate_bytes, gate_sha = _json_snapshot(release_gate)
    freeze, _, freeze_sha = _json_snapshot(freeze_manifest)
    artifact_input, artifact_bytes, artifact_sha = _json_snapshot(
        artifacts / "manifest.json"
    )
    model_id, revision = record.get("model_id"), record.get("model_revision")
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ValueError("Model ID must be a Decision 2.0 release name")
    if (
        not isinstance(revision, str)
        or not revision
        or not isinstance(score_key, str)
        or not score_key
    ):
        raise ValueError("Model revision and score key are required")
    _rights(record, model_id, revision)
    architecture = record.get("architecture")
    external = architecture == EXTERNAL_ADAPTER
    if external:
        if base_source is None or adapter_source_parity_receipt is None:
            raise ValueError("External PEFT release needs base and source parity")
    elif base_source is not None or adapter_source_parity_receipt is not None:
        raise ValueError("External PEFT inputs are invalid for this architecture")
    files = _inventory(model_dir, allow_adapter_metadata=external)
    if record.get("model_files_sha256") != files:
        raise ValueError("Model files differ from the frozen package record")
    _profile(model_dir, architecture, files)
    external_manifest = None
    source_parity_sha = None
    checkpoint_files = None
    if external:
        assert base_source is not None and adapter_source_parity_receipt is not None
        external_manifest, count, checkpoint_files = _external_adapter_contract(
            model_dir, base_source, record, files
        )
        source_parity_sha = _adapter_source_parity(
            adapter_source_parity_receipt,
            external_manifest,
            files["MODEL_MANIFEST.json"],
        )
    else:
        count = _parameter_count(
            model_dir,
            record.get("active_weight_files"),
            record.get("support_weight_files"),
            record.get("non_parameter_tensors"),
            files,
        )
    if (
        record.get("parameter_count") != count
        or type(record.get("parameter_count")) is not int
    ):
        raise ValueError("Actual safetensors parameter count differs from declaration")
    if not _size_compatible(count, model_id):
        raise ValueError(
            "Actual parameter count differs materially from model size name"
        )
    native_sha = _native_identity(model_dir, record, files, external_manifest)
    calibration_name = record.get("calibration_file")
    permitted_calibration = (
        {"calib.json"}
        if architecture == "qwen3.5-semif"
        else (
            {"calibration.json", "calib.json"}
            if architecture == "encoder-decision"
            else {"calibration.json"}
        )
    )
    if calibration_name not in permitted_calibration:
        raise ValueError("Native CAL filename differs from architecture contract")
    if record.get("calibration_sha256") != files.get(calibration_name):
        raise ValueError("Native CAL artifact differs from provenance")
    artifacts_manifest, row, roster_sha = _release_artifacts(
        artifacts,
        arena_rank,
        public_rank,
        model_id,
        revision,
        score_key,
        artifact_input,
    )
    if not math.isclose(row["size_b"], count / 1e9, rel_tol=0, abs_tol=1e-9):
        raise ValueError("JevArena parameter count differs from actual package weights")
    model_files_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    score_binding = _score_inputs(
        score_inputs,
        row,
        native_sha,
        model_id,
        revision,
        files[calibration_name],
        record["native_adapter_version"],
        files["MODEL_MANIFEST.json"] if external else None,
        checkpoint_files,
    )
    adapter_shas = {entry["adapter_sha256"] for entry in score_binding.values()}
    if len(adapter_shas) != 1:
        raise ValueError("Same-panel native runs used different adapter bytes")
    candidate = {
        "model_id": model_id,
        "model_revision": revision,
        "native_model_sha256": native_sha,
        "model_files_sha256": model_files_digest,
        "calibration_sha256": files[calibration_name],
        "adapter_version": record["native_adapter_version"],
        "adapter_sha256": next(iter(adapter_shas)),
    }
    _parity(
        parity,
        model_id,
        revision,
        native_sha,
        model_files_digest,
        files[calibration_name],
    )
    _gate(
        gate,
        record_sha=record_sha,
        parity_sha=parity_sha,
        artifact_sha=artifact_sha,
        native_sha=native_sha,
        model_files_digest=model_files_digest,
        model_id=model_id,
        revision=revision,
    )
    external_hashes = _external_evidence(
        record,
        gate,
        provenance_inputs,
        freeze_manifest,
        gate_evidence,
        freeze=freeze,
        freeze_sha=freeze_sha,
        panel=artifacts_manifest["panel_sha256"],
        coverage=row["coverage"],
        candidate=candidate,
        roster_sha=roster_sha,
        score_binding=score_binding,
    )
    # Records copied to the public repository contain only reviewed, screened text.
    for path, payload in (
        (package_record, record_bytes),
        (parity_receipt, parity_bytes),
        (release_gate, gate_bytes),
    ):
        _public_text(payload.decode("utf-8"), path.name)
    sticker = Path(__file__).with_name("decision-2-sticker-chibi-v4.png")
    if not sticker.is_file() or sticker.is_symlink():
        raise ValueError("Family sticker asset is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in files:
            target = temporary / "native" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_dir / name, target)
            if sha_file(target) != files[name]:
                raise ValueError("Model file changed during packaging")
        if architecture in {"qwen3.5-decision-head", "qwen3.8-decision-head"}:
            runtime = _qwen_runtime_contract(temporary / "native", native_sha)
            runtime["files_sha256"] = {
                name: sha_file(temporary / "native" / name) for name in files
            }
            (temporary / "native" / "MODEL_MANIFEST.json").write_text(
                json.dumps(
                    runtime,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            _verify_qwen_runtime(temporary / "native")
        elif external:
            assert base_source is not None
            from .adapter_bundle import _verify_staged_runtime

            _verify_staged_runtime(temporary / "native", base_source)
        (temporary / "card-artifacts").mkdir()
        for name in (*ARTIFACTS, "manifest.json"):
            target = temporary / "card-artifacts" / name
            if name == "manifest.json":
                target.write_bytes(artifact_bytes)
                expected_hash = artifact_sha
            else:
                shutil.copyfile(artifacts / name, target)
                expected_hash = artifact_input["artifacts_sha256"][name]
            if sha_file(target) != expected_hash:
                raise ValueError(f"Release artifact changed during packaging: {name}")
            if name in ARTIFACTS:
                shutil.copyfile(target, temporary / name)
        for payload, digest, name in (
            (record_bytes, record_sha, "release-record.json"),
            (parity_bytes, parity_sha, "native-parity.json"),
            (gate_bytes, gate_sha, "release-gate.json"),
        ):
            target = temporary / name
            target.write_bytes(payload)
            if sha_file(target) != digest:
                raise ValueError(f"Reviewed release receipt changed in package: {name}")
        shutil.copyfile(sticker, temporary / sticker.name)
        (temporary / "README.md").write_text(
            _card(
                model_id,
                record,
                row,
                count,
                (temporary / "score-table.md").read_text(encoding="utf-8"),
            ),
            encoding="utf-8",
        )
        public_files = {
            path.relative_to(temporary).as_posix(): sha_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "bundle_version": VERSION,
            "model_id": model_id,
            "model_revision": revision,
            "architecture": architecture,
            "parameter_count": count,
            "native_model_sha256": native_sha,
            "model_files_sha256": model_files_digest,
            "artifact_manifest_sha256": artifact_sha,
            "panel_sha256": artifacts_manifest["panel_sha256"],
            "package_record_sha256": record_sha,
            "parity_receipt_sha256": parity_sha,
            "release_gate_sha256": gate_sha,
            "score_inputs_sha256": score_binding,
            "files_sha256": public_files,
        }
        if external:
            assert external_manifest is not None and source_parity_sha is not None
            manifest["external_base"] = external_manifest["base"]
            manifest["adapter_manifest_sha256"] = files["MODEL_MANIFEST.json"]
            manifest["adapter_source_parity_sha256"] = source_parity_sha
        (temporary / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        verify(temporary, base_source=base_source)
        for name, path, digest in (
            ("release record", package_record, record_sha),
            ("native parity", parity_receipt, parity_sha),
            ("release gate", release_gate, gate_sha),
            ("artifact manifest", artifacts / "manifest.json", artifact_sha),
        ):
            if path.is_symlink() or not path.is_file() or sha_file(path) != digest:
                raise ValueError(f"Validated {name} changed while packaging")
        for name, path in {
            **{f"gate:{key}": value for key, value in gate_evidence.items()},
            **provenance_inputs,
            "pretest_freeze": freeze_manifest,
        }.items():
            if (
                path.is_symlink()
                or not path.is_file()
                or sha_file(path) != external_hashes[name]
            ):
                raise ValueError(f"External evidence changed while packaging: {name}")
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify(root: Path, *, base_source: Path | None = None) -> dict[str, Any]:
    """Check all staged public bytes and their model/artifact binding on CPU."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Publication package must be a regular directory")
    manifest = _object(root / "PACKAGE_MANIFEST.json")
    if manifest.get("bundle_version") != VERSION:
        raise ValueError("Unknown JevArena package version")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Package file inventory is missing")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Published package file inventory has changed")
    external = manifest.get("architecture") == EXTERNAL_ADAPTER
    if external:
        native = adapter_runtime._inventory(root / "native", ignore_bytecode=True)
        actual = {f"native/{name}": digest for name, digest in native.items()}
        actual.update(
            {
                path.relative_to(root).as_posix(): sha_file(path)
                for path in root.rglob("*")
                if path.is_file()
                and not path.is_relative_to(root / "native")
                and path != root / "PACKAGE_MANIFEST.json"
            }
        )
    else:
        if base_source is not None:
            raise ValueError("External base source was supplied to a full package")
        actual = {
            path.relative_to(root).as_posix(): sha_file(path)
            for path in root.rglob("*")
            if path.is_file() and path != root / "PACKAGE_MANIFEST.json"
        }
    if actual != expected:
        raise ValueError("Published package file inventory has changed")
    model_files = {
        name.removeprefix("native/"): digest
        for name, digest in actual.items()
        if name.startswith("native/")
        and (external or name != "native/MODEL_MANIFEST.json")
    }
    digest = hashlib.sha256(
        json.dumps(model_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("model_files_sha256"):
        raise ValueError("Native model identity changed inside publication package")
    if actual.get("card-artifacts/manifest.json") != manifest.get(
        "artifact_manifest_sha256"
    ):
        raise ValueError("Card artifact manifest changed")
    record = _object(root / "release-record.json")
    parity = _object(root / "native-parity.json")
    gate = _object(root / "release-gate.json")
    artifacts = _object(root / "card-artifacts/manifest.json")
    model_id, revision = manifest.get("model_id"), manifest.get("model_revision")
    count = manifest.get("parameter_count")
    if (
        not isinstance(model_id, str)
        or not MODEL_ID.fullmatch(model_id)
        or not isinstance(revision, str)
        or not revision
        or type(count) is not int
        or count <= 0
        or not _size_compatible(count, model_id)
        or record.get("architecture") != manifest.get("architecture")
        or record.get("parameter_count") != count
        or record.get("model_files_sha256") != model_files
        or not isinstance(record.get("native_identity"), dict)
        or record["native_identity"].get("sha256")
        != manifest.get("native_model_sha256")
        or actual.get("release-record.json") != manifest.get("package_record_sha256")
        or actual.get("native-parity.json") != manifest.get("parity_receipt_sha256")
        or actual.get("release-gate.json") != manifest.get("release_gate_sha256")
    ):
        raise ValueError("Publication manifest, record or receipt binding changed")
    _rights(record, model_id, revision)
    calibration = record.get("calibration_file")
    calibration_sha = actual.get(f"native/{calibration}")
    if calibration_sha is None or calibration_sha != record.get("calibration_sha256"):
        raise ValueError("Native CAL differs from release record")
    if (
        artifacts.get("publication_version") != ARTIFACT_VERSION
        or artifacts.get("phase") != "release"
        or artifacts.get("panel_sha256") != manifest.get("panel_sha256")
        or not isinstance(artifacts.get("artifacts_sha256"), dict)
        or set(artifacts["artifacts_sha256"]) != set(ARTIFACTS)
        or any(
            actual.get(f"card-artifacts/{name}") != artifacts["artifacts_sha256"][name]
            or actual.get(name) != artifacts["artifacts_sha256"][name]
            for name in ARTIFACTS
        )
    ):
        raise ValueError("Publication artifact manifest or duplicated artifact changed")
    _parity(
        parity,
        model_id,
        revision,
        manifest["native_model_sha256"],
        digest,
        calibration_sha,
    )
    _gate(
        gate,
        record_sha=actual["release-record.json"],
        parity_sha=actual["native-parity.json"],
        artifact_sha=actual["card-artifacts/manifest.json"],
        native_sha=manifest["native_model_sha256"],
        model_files_digest=digest,
        model_id=model_id,
        revision=revision,
    )
    if manifest.get("architecture") in {
        "qwen3.5-decision-head",
        "qwen3.8-decision-head",
    }:
        _verify_qwen_runtime(root / "native")
    elif external:
        if base_source is not None:
            from .adapter_bundle import _verify_staged_runtime

            _verify_staged_runtime(root / "native", base_source)
        inner = adapter_runtime.verify_bundle(root / "native")
        base = inner["base"]
        if (
            manifest.get("external_base") != base
            or manifest.get("adapter_manifest_sha256")
            != actual.get("native/MODEL_MANIFEST.json")
            or manifest.get("native_model_sha256") != inner["model_sha256"]
            or manifest.get("parameter_count") != inner["parameter_count"]
            or record.get("base_model", {}).get("id") != base["repo_id"]
            or record.get("base_model", {}).get("revision") != base["revision"]
            or record.get("model_id") != inner["model_id"]
        ):
            raise ValueError("External PEFT base, adapter or release record changed")
        _sha(manifest.get("adapter_source_parity_sha256"), "source parity")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _object(args.config)
    required = {
        "model_dir",
        "artifacts",
        "arena_rank",
        "public_rank",
        "package_record",
        "parity_receipt",
        "release_gate",
        "provenance_inputs",
        "freeze_manifest",
        "gate_evidence",
        "score_inputs",
        "score_key",
    }
    if not required <= set(config) or set(config) - required - {
        "base_source",
        "adapter_source_parity_receipt",
    }:
        raise ValueError("Release packaging config has missing or unknown fields")
    base = args.config.parent

    def source(name: str) -> Path:
        path = Path(name)
        return path if path.is_absolute() else base / path

    score_inputs = {
        family: {name: source(path) for name, path in values.items()}
        for family, values in config["score_inputs"].items()
    }
    provenance_inputs = {
        name: source(path) for name, path in config["provenance_inputs"].items()
    }
    gate_evidence = {
        name: source(path) for name, path in config["gate_evidence"].items()
    }
    manifest = assemble(
        model_dir=source(config["model_dir"]),
        artifacts=source(config["artifacts"]),
        arena_rank=source(config["arena_rank"]),
        public_rank=source(config["public_rank"]),
        package_record=source(config["package_record"]),
        parity_receipt=source(config["parity_receipt"]),
        release_gate=source(config["release_gate"]),
        provenance_inputs=provenance_inputs,
        freeze_manifest=source(config["freeze_manifest"]),
        gate_evidence=gate_evidence,
        score_inputs=score_inputs,
        score_key=config["score_key"],
        output=args.output,
        base_source=(
            source(config["base_source"]) if "base_source" in config else None
        ),
        adapter_source_parity_receipt=(
            source(config["adapter_source_parity_receipt"])
            if "adapter_source_parity_receipt" in config
            else None
        ),
    )
    print(
        json.dumps(
            {
                "model_id": manifest["model_id"],
                "parameter_count": manifest["parameter_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
