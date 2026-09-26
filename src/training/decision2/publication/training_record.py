"""Bind public training disclosures to a completed run and its CAL report.

The source-data builder manifest and run receipts stay private. Only the
reviewed disclosure and a narrow hash/count summary enter the model bundle.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from training.model.data import canonical
from training.model.infer import MODEL_ROOT_FILES

from .rights_gate import verify_rights

VERSION = "decision2-public-training-record/1"
HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")
REVISION = re.compile(r"[a-f0-9]{40}\Z")
CHECKPOINT = re.compile(r"checkpoint-[0-9]{7}\Z")


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Training receipt must be a regular file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Training receipt must be a JSON object: {path.name}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Training provenance {label} differs from frozen CAL or source"
        )


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 500
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"Training record {label} needs one nonempty public-text line")
    return value.strip()


def _notes(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 30:
        raise ValueError(f"Training record {label} needs a nonempty list")
    return [_text(item, label) for item in value]


def _url(value: Any) -> str:
    value = _text(value, "source URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(char in value for char in " <>[]()\\")
    ):
        raise ValueError("Training record source URL must be a public HTTPS URL")
    host = parsed.hostname.lower()
    if host in {
        "localhost",
        "example.com",
        "example.org",
        "example.net",
    } or host.endswith((".local", ".internal", ".localhost")):
        raise ValueError("Training record source URL must be public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Training record source URL must be public")
    return value


def _data_counts(value: Any, label: str) -> dict[str, int]:
    if (
        not isinstance(value, dict)
        or not value
        or any(
            not isinstance(key, str) or not key or type(count) is not int or count < 0
            for key, count in value.items()
        )
    ):
        raise ValueError(f"Training data manifest lacks valid {label} counts")
    return value


def bind_training_record(
    *,
    record_path: Path,
    run_dir: Path,
    data_manifest_path: Path,
    calibration: dict[str, Any],
    merged_receipt: dict[str, Any],
    source_model_sha256: str,
    rights_attestation_path: Path | None,
    license_id: str,
) -> dict[str, Any]:
    """Return a public summary only after exact run/data/model hash checks."""
    record = _read(record_path)
    data = _read(data_manifest_path)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("Completed training run must be a regular directory")
    if (
        set(record)
        != {
            "record_version",
            "initialization",
            "sources",
            "known_overlap",
            "evaluation_interpretation",
            "limitations",
        }
        or record["record_version"] != VERSION
    ):
        raise ValueError("Unknown or incomplete public training record")

    best_path, complete_path, provenance_path = (
        run_dir / "BEST.json",
        run_dir / "COMPLETE.json",
        run_dir / "provenance.json",
    )
    best, complete, provenance = (
        _read(path) for path in (best_path, complete_path, provenance_path)
    )
    for key, path in (
        ("best_sha256", best_path),
        ("complete_sha256", complete_path),
        ("provenance_sha256", provenance_path),
    ):
        _same(calibration.get(key), _sha(path), key)
    selected = best.get("checkpoint")
    if not isinstance(selected, str) or not CHECKPOINT.fullmatch(selected):
        raise ValueError("BEST names an invalid selected checkpoint")
    if (
        complete.get("status") != "complete"
        or complete.get("best") != selected
        or calibration.get("selected_checkpoint") != selected
    ):
        raise ValueError(
            "Training run is incomplete or CAL used another BEST checkpoint"
        )
    checkpoint_path = run_dir / selected
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise ValueError("Selected training checkpoint is unavailable")
    checkpoint = _read(checkpoint_path / "checkpoint.json")
    if checkpoint.get("complete") is not True or checkpoint.get("step") != int(
        selected.split("-")[-1]
    ):
        raise ValueError("Selected training checkpoint receipt is incomplete")
    metadata = _read(checkpoint_path / "decision_config.json")
    contract = provenance.get("contract")
    model_source = provenance.get("model_source")
    if not isinstance(contract, dict) or not isinstance(model_source, dict):
        raise ValueError("Run provenance lacks the training contract and model source")
    _same(contract.get("model_source"), model_source, "initialization source")
    if metadata.get("checkpoint_format") != "peft-lora/1" or not isinstance(
        metadata.get("lora"), dict
    ):
        raise ValueError(
            "Selected checkpoint is not a LoRA source for the merged weights"
        )
    _same(
        metadata["lora"].get("source_fingerprint"),
        model_source,
        "selected checkpoint initialization",
    )
    source_files = model_source.get("files_sha256")
    if (
        not isinstance(source_files, dict)
        or not source_files
        or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or not SHA.fullmatch(value)
            for name, value in source_files.items()
        )
    ):
        raise ValueError("Run provenance has invalid source-file hashes")
    source_digest = hashlib.sha256(canonical(source_files).encode("utf-8")).hexdigest()
    _same(
        calibration.get("initialization_source_sha256"),
        source_digest,
        "initialization source hash",
    )
    if calibration.get("loaded_source_sha256") is not None:
        _same(calibration["loaded_source_sha256"], source_digest, "loaded source hash")
    receipt_files = merged_receipt.get("source_model_files_sha256")
    if not isinstance(receipt_files, dict):
        raise ValueError("Merge receipt lacks original source-model files")
    receipt_digest = hashlib.sha256(
        canonical(receipt_files).encode("utf-8")
    ).hexdigest()
    _same(receipt_digest, source_model_sha256, "merged source-model files")
    _same(
        {
            name.removeprefix("source/"): digest
            for name, digest in receipt_files.items()
            if name.startswith("source/")
        },
        source_files,
        "merged initialization files",
    )
    model_paths = [
        path
        for path in checkpoint_path.iterdir()
        if path.is_file() and path.name in MODEL_ROOT_FILES
    ]
    adapter_dir = checkpoint_path / "adapter"
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise ValueError("Selected LoRA checkpoint lacks a regular adapter directory")
    model_paths.extend(
        path
        for path in adapter_dir.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}
    )
    if any(path.is_symlink() for path in model_paths):
        raise ValueError("Selected checkpoint contains a symlinked model file")
    actual_checkpoint_files = {
        str(path.relative_to(checkpoint_path)): _sha(path) for path in model_paths
    }
    _same(
        actual_checkpoint_files,
        {
            name.removeprefix("checkpoint/"): digest
            for name, digest in receipt_files.items()
            if name.startswith("checkpoint/")
        },
        "selected checkpoint weight files",
    )
    checkpoint_digest = hashlib.sha256(
        canonical(actual_checkpoint_files).encode("utf-8")
    ).hexdigest()
    _same(
        calibration.get("checkpoint_sha256"),
        checkpoint_digest,
        "selected checkpoint file hash",
    )
    _same(
        merged_receipt.get("source_model_sha256"),
        source_model_sha256,
        "merged source-model identity",
    )
    _same(calibration.get("model_sha256"), source_model_sha256, "CAL model identity")

    data_sha = contract.get("data_sha256")
    if not isinstance(data_sha, dict) or set(data_sha) != {"train", "select", "cal"}:
        raise ValueError(
            "Training record v1 requires frozen TRAIN, SELECT and CAL without replay"
        )
    if any(
        not isinstance(value, str) or not SHA.fullmatch(value)
        for value in data_sha.values()
    ):
        raise ValueError("Run provenance has invalid partition hashes")
    _same(calibration.get("cal_sha256"), data_sha["cal"], "CAL partition")
    _same(
        calibration.get("inference", {}).get("max_length"),
        contract.get("max_length"),
        "inference context length",
    )
    outputs = data.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Training data manifest lacks output receipts")
    matched = {}
    for role, digest in data_sha.items():
        candidates = [
            (name, value)
            for name, value in outputs.items()
            if isinstance(value, dict) and value.get("sha256") == digest
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Training data manifest does not uniquely bind {role} bytes"
            )
        matched[role] = candidates[0][1]
    for role, expected in (
        ("train", provenance.get("train_examples")),
        ("select", provenance.get("select_examples")),
        ("cal", provenance.get("cal_examples_audited_only")),
    ):
        rows = matched[role].get("rows")
        if type(rows) is not int or rows < 1 or rows != expected:
            raise ValueError(
                f"Training data manifest {role} row count differs from run"
            )
    counts = data.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Training data manifest lacks source and task-type counts")
    source_counts = _data_counts(counts.get("source"), "source")
    type_counts = _data_counts(counts.get("task_type"), "task-type")
    if (
        sum(source_counts.values()) != matched["train"]["rows"]
        or sum(type_counts.values()) != matched["train"]["rows"]
    ):
        raise ValueError(
            "Training data manifest source/type counts differ from TRAIN rows"
        )

    initialization = record["initialization"]
    if (
        not isinstance(initialization, dict)
        or set(initialization)
        != {"model_id", "revision", "source_name", "license_status"}
        or not isinstance(initialization["model_id"], str)
        or not HF_ID.fullmatch(initialization["model_id"])
        or not isinstance(initialization["revision"], str)
        or not REVISION.fullmatch(initialization["revision"])
    ):
        raise ValueError(
            "Training record needs a pinned public initialization identity"
        )
    _same(
        initialization["source_name"],
        model_source.get("source_name"),
        "declared initialization source name",
    )
    _text(initialization["license_status"], "initialization license status")
    sources = record["sources"]
    if not isinstance(sources, list) or not sources or len(sources) > 100:
        raise ValueError("Training record requires dataset source disclosures")
    documented = {}
    for source in sources:
        if not isinstance(source, dict) or set(source) != {
            "source",
            "rows",
            "url",
            "license_status",
            "attribution",
        }:
            raise ValueError(
                "Training record source requires exact source/count/rights fields"
            )
        name = _text(source["source"], "source")
        if name in documented or type(source["rows"]) is not int or source["rows"] < 1:
            raise ValueError("Training record has duplicate or invalid source counts")
        _url(source["url"])
        _text(source["license_status"], "source license status")
        _text(source["attribution"], "source attribution")
        documented[name] = source["rows"]
    _same(documented, source_counts, "declared source counts")
    overlap = _notes(record["known_overlap"], "known overlap")
    interpretation = _notes(
        record["evaluation_interpretation"], "evaluation interpretation"
    )
    limitations = _notes(record["limitations"], "limitations")
    builder_limitations = _notes(data.get("limitations"), "builder limitations")
    rights = verify_rights(
        data=data,
        data_manifest_path=data_manifest_path,
        run_provenance_path=provenance_path,
        partition_sha=data_sha,
        partition_rows={role: matched[role]["rows"] for role in data_sha},
        source_counts=source_counts,
        attestation_path=rights_attestation_path,
        license_id=license_id,
    )
    if (
        type(contract.get("planned_updates")) is not int
        or contract["planned_updates"] < 1
    ):
        raise ValueError("Run provenance lacks planned optimizer steps")
    if (
        complete.get("step") != contract["planned_updates"]
        or complete.get("planned_updates") != contract["planned_updates"]
    ):
        raise ValueError("Completed run steps differ from frozen training plan")
    if checkpoint["step"] > complete["step"]:
        raise ValueError("Selected checkpoint step exceeds the completed run")
    if contract.get("train_mode") != "lora" or not isinstance(
        contract.get("lora"), dict
    ):
        raise ValueError("This bundle format requires a completed LoRA training run")
    for key in ("epochs", "microbatch", "accumulation", "max_length", "seed"):
        if type(contract.get(key)) is not int or contract[key] < 1:
            raise ValueError(f"Run provenance has invalid {key}")
    for key in ("brier_weight", "head_lr", "weight_decay", "warmup_ratio"):
        value = contract.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Run provenance has invalid {key}")
    _text(contract.get("objective"), "optimization objective")
    _text(provenance.get("precision"), "training precision")
    _text(best.get("selection"), "checkpoint selection rule")
    lora = contract["lora"]
    for key in ("rank", "alpha"):
        if type(lora.get(key)) is not int or lora[key] < 1:
            raise ValueError(f"Run provenance has invalid LoRA {key}")
    for key in ("dropout", "lr"):
        value = lora.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Run provenance has invalid LoRA {key}")
    return {
        "provenance_version": "decision2-verified-public-training-provenance/1",
        "declaration": record,
        "verification": {
            "training_record_sha256": _sha(record_path),
            "data_manifest_sha256": _sha(data_manifest_path),
            "data_manifest_version": _text(
                data.get("schema_version"), "data manifest version"
            ),
            "run_receipts_sha256": {
                key: _sha(path)
                for key, path in (
                    ("best", best_path),
                    ("complete", complete_path),
                    ("provenance", provenance_path),
                )
            },
            "source_model_sha256": source_model_sha256,
            "source_files_sha256": source_digest,
            "selected_checkpoint": selected,
            "selected_step": checkpoint["step"],
            "partition_sha256": data_sha,
            "partition_rows": {role: matched[role]["rows"] for role in data_sha},
            "source_counts": source_counts,
            "task_type_counts": type_counts,
        },
        "optimization": {
            "train_mode": contract["train_mode"],
            "initialization_kind": contract.get("init_kind"),
            "objective": contract.get("objective"),
            "brier_weight": contract.get("brier_weight"),
            "epochs": contract.get("epochs"),
            "planned_updates": contract["planned_updates"],
            "microbatch": contract.get("microbatch"),
            "accumulation": contract.get("accumulation"),
            "max_length": contract["max_length"],
            "seed": contract.get("seed"),
            "lora": contract["lora"],
            "head_lr": contract.get("head_lr"),
            "weight_decay": contract.get("weight_decay"),
            "warmup_ratio": contract.get("warmup_ratio"),
            "precision": provenance.get("precision"),
            "selection": best.get("selection"),
        },
        "calibration": {
            "fit_split": calibration.get("fit_split"),
            "selection_policy": calibration.get("selection_policy"),
            "cal_rows": matched["cal"]["rows"],
            "temperature_by_type": calibration.get("temperature_by_type"),
        },
        "builder_limitations": builder_limitations,
        "rights": rights,
        "known_overlap": overlap,
        "evaluation_interpretation": interpretation,
        "limitations": limitations,
    }
