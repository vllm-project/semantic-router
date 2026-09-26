"""Fail-closed JevArena release packager for already self-contained models.

This is a byte-integrity and process gate, not a GPU parity runner. The input
model directory must already run using its embedded native inference code.
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
from pathlib import Path
from typing import Any

from .generate_arena import ARTIFACTS, matched_models
from .generate_arena import VERSION as ARTIFACT_VERSION

VERSION = "decision2-jevarena-self-contained-bundle/1"
RECORD_VERSION = "decision2-release-package-record/1"
PARITY_VERSION = "decision2-native-package-parity/1"
GATE_VERSION = "decision2-jevarena-release-gate/1"
MODEL_ID = re.compile(r"llm-semantic-router/dev-2\.0-(?:0\.6b|0\.8b|2b|4b|8b|9b|27b)\Z")
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
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSON file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


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


def _inventory(root: Path) -> dict[str, str]:
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
        }:
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
    if architecture in {"qwen3.5-decision-head", "qwen3.8-decision-head"}:
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
        raise ValueError(
            f"Self-contained {architecture} package lacks {sorted(required - names)}"
        )
    if architecture != "encoder-decision" and any(
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


def _native_identity(root: Path, record: dict[str, Any], files: dict[str, str]) -> str:
    identity = record.get("native_identity")
    if not isinstance(identity, dict) or set(identity) != {"scheme", "sha256", "file"}:
        raise ValueError("Native model identity declaration is incomplete")
    expected = _sha(identity["sha256"], "native model identity")
    scheme, name = identity["scheme"], identity["file"]
    if scheme == "qwen-checkpoint-fingerprint":
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _object(artifacts / "manifest.json")
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
    return manifest, row


def _score_inputs(
    paths: dict[str, dict[str, Path]],
    row: dict[str, Any],
    native_sha: str,
    model_id: str,
    revision: str,
    calibration_sha: str,
    adapter_version: str,
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
        binding[family] = {
            "score_sha256": score_sha,
            "predictions_sha256": prediction_sha,
            "native_manifest_sha256": native_sha_file,
        }
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


def _external_evidence(
    record: dict[str, Any],
    gate: dict[str, Any],
    provenance_inputs: dict[str, Path],
    freeze_manifest: Path,
    gate_evidence: dict[str, Path],
) -> None:
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
        if sha_file(path) != wanted:
            raise ValueError(f"External evidence changed after review: {name}")


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
    overlap = (
        "\n".join(f"- {_md(value)}" for value in record["known_overlap"])
        or "- No overlap declared in the reviewed record."
    )
    return f"""---
license: {record['license_id']}
{('license_name: noncommercial-research-terms' + chr(10)) if record['license_id'] == 'other' else ''}base_model: {record['base_model']['id']}
tags:
- decision-model
- typed-decision
- jevarena
---

![Decision 2.0 chibi pixel-mosaic owl](decision-2-sticker-chibi-v2.png)

# {model_id}

Native architecture: `{record['architecture']}`. Actual model parameters:
**{count:,}**. Source model: [{record['base_model']['id']}](https://huggingface.co/{record['base_model']['id']})
at immutable revision `{record['base_model']['revision']}`. Native runtime and
calibration are included in this package; its exact invocation and limits are
described by the bundled runtime and `PACKAGE_MANIFEST.json`.

## Same-panel release evaluation

JevArena rank **#{row['rank']}**, six-axis score **{row['score']:.2f}** on the
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

TRAIN {record['training']['train_rows']:,}; SELECT {record['training']['select_rows']:,};
CAL {record['training']['cal_rows']:,}. Selection policy:
{_md(record['training']['selection_policy'])}. Scope: `{rights['scope']}`.
No raw upstream text or benchmark labels are bundled.

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
    model_dir, artifacts = model_dir.resolve(strict=True), artifacts.resolve(
        strict=True
    )
    output = output.resolve()
    if (
        output.exists()
        or output.is_relative_to(model_dir)
        or output.is_relative_to(artifacts)
    ):
        raise FileExistsError("Output exists or is inside an input directory")
    record, parity, gate = (
        _object(path) for path in (package_record, parity_receipt, release_gate)
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
    files = _inventory(model_dir)
    if record.get("model_files_sha256") != files:
        raise ValueError("Model files differ from the frozen package record")
    architecture = record.get("architecture")
    _profile(model_dir, architecture, files)
    active, support, excluded = (
        record.get("active_weight_files"),
        record.get("support_weight_files"),
        record.get("non_parameter_tensors"),
    )
    count = _parameter_count(model_dir, active, support, excluded, files)
    if (
        record.get("parameter_count") != count
        or type(record.get("parameter_count")) is not int
    ):
        raise ValueError("Actual safetensors parameter count differs from declaration")
    suffix = model_id.rsplit("-", 1)[-1]
    if abs(count / 1e9 - float(suffix[:-1])) / float(suffix[:-1]) > 0.25:
        raise ValueError(
            "Actual parameter count differs materially from model size name"
        )
    native_sha = _native_identity(model_dir, record, files)
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
    artifacts_manifest, row = _release_artifacts(
        artifacts, arena_rank, public_rank, model_id, revision, score_key
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
    )
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
        record_sha=sha_file(package_record),
        parity_sha=sha_file(parity_receipt),
        artifact_sha=sha_file(artifacts / "manifest.json"),
        native_sha=native_sha,
        model_files_digest=model_files_digest,
        model_id=model_id,
        revision=revision,
    )
    _external_evidence(record, gate, provenance_inputs, freeze_manifest, gate_evidence)
    # Records copied to the public repository contain only reviewed, screened text.
    for path in (package_record, parity_receipt, release_gate):
        _public_text(path.read_text(encoding="utf-8"), path.name)
    sticker = Path(__file__).with_name("decision-2-sticker-chibi-v2.png")
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
        (temporary / "card-artifacts").mkdir()
        for name in (*ARTIFACTS, "manifest.json"):
            shutil.copyfile(artifacts / name, temporary / "card-artifacts" / name)
            if name in ARTIFACTS:
                shutil.copyfile(artifacts / name, temporary / name)
        for source, name in (
            (package_record, "release-record.json"),
            (parity_receipt, "native-parity.json"),
            (release_gate, "release-gate.json"),
        ):
            shutil.copyfile(source, temporary / name)
        shutil.copyfile(sticker, temporary / sticker.name)
        (temporary / "README.md").write_text(
            _card(
                model_id,
                record,
                row,
                count,
                (artifacts / "score-table.md").read_text(encoding="utf-8"),
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
            "artifact_manifest_sha256": sha_file(artifacts / "manifest.json"),
            "panel_sha256": artifacts_manifest["panel_sha256"],
            "package_record_sha256": sha_file(package_record),
            "parity_receipt_sha256": sha_file(parity_receipt),
            "release_gate_sha256": sha_file(release_gate),
            "score_inputs_sha256": score_binding,
            "files_sha256": public_files,
        }
        (temporary / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        verify(temporary)
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify(root: Path) -> dict[str, Any]:
    """Check all staged public bytes and their model/artifact binding on CPU."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Publication package must be a regular directory")
    manifest = _object(root / "PACKAGE_MANIFEST.json")
    if manifest.get("bundle_version") != VERSION:
        raise ValueError("Unknown JevArena package version")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Package file inventory is missing")
    actual = {
        path.relative_to(root).as_posix(): sha_file(path)
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != "PACKAGE_MANIFEST.json"
    }
    if any(path.is_symlink() for path in root.rglob("*")) or actual != expected:
        raise ValueError("Published package file inventory has changed")
    model_files = {
        name.removeprefix("native/"): digest
        for name, digest in actual.items()
        if name.startswith("native/") and name != "native/MODEL_MANIFEST.json"
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
    if manifest.get("architecture") in {
        "qwen3.5-decision-head",
        "qwen3.8-decision-head",
    }:
        _verify_qwen_runtime(root / "native")
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
    if set(config) != required:
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
