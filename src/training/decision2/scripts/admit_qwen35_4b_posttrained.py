"""Version 2 bounded admission for the official Qwen3.5-4B Posttrained source.

``prepare`` and ``compare-zero`` are CPU-only. ``zero-a``, ``zero-b``,
``one`` and ``reload`` are GPU stages and must not run before independent
review of the private lock. All files, including SELECT predictions, stay in
the owner's private experiment directory. No formal benchmark is accepted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.preflight_qwen35_4b_posttrained_cpu import (
    POSTTRAINED_REVISION,
    sha256_file,
)
from training.model.data import check_partition_isolation, load_partition
from training.model.source import source_fingerprint

SCHEMA = "decision2-qwen35-4b-posttrained-admission/2"
SOURCE_ID = "Qwen/Qwen3.5-4B"
IMAGE_ID = "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
RUNTIME_PYTHON = "/usr/bin/python"
TRANSFORMERS_VERSION = "5.17.0"
TORCH_VERSION = "2.12.0+git6bbd260"
CPU_AUDIT_SHA256 = "0c45417d4cc8f3e4c32edc55aec3c230e0922476500acc1fabc3b5c2151ac82b"
DATA_SHA256 = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
SHARD_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
COUNT = {"train": 7455, "select": 700, "cal": 700}
TYPES = {"choice": 320, "noul": 290, "score": 90}
ZERO_SECONDS = 540
ONE_SECONDS = 360
RELOAD_SECONDS = 180
MAX_P99_DRIFT = 0.005
MAX_DRIFT = 0.02
PROTOCOL = {
    "zero_starts": 2,
    "zero_rows": 700,
    "one_updates": 1,
    "reload_rows": 32,
    "max_length": 8192,
    "zero_max_seconds_each": ZERO_SECONDS,
    "one_max_seconds": ONE_SECONDS,
    "reload_max_seconds": RELOAD_SECONDS,
    "categorical_changes": 0,
    "probability_p99_max": MAX_P99_DRIFT,
    "probability_absolute_max": MAX_DRIFT,
    "seed": 20260926,
    "init_kind": "posttrained",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "objective": "ce_brier",
    "brier_weight": 0.5,
    "runtime_python": RUNTIME_PYTHON,
    "transformers_version": TRANSFORMERS_VERSION,
    "torch_version": TORCH_VERSION,
    "cpu_full_weight_source_load": True,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_hashes() -> dict[str, str]:
    root = _repo_root()
    names = (
        "scripts/admit_qwen35_4b_posttrained.py",
        "scripts/preflight_qwen35_4b_posttrained_cpu.py",
        "training/model/data.py",
        "training/model/decision_model.py",
        "training/model/lora.py",
        "training/model/loss.py",
        "training/model/plan.py",
        "training/model/source.py",
        "training/model/train.py",
    )
    return {name: sha256_file(root / name) for name in names}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _private_root(root: Path) -> Path:
    root = root.resolve(strict=True)
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("Experiment directory must exist and have mode 0700")
    return root


def _validate_runtime(
    python: str, transformers_version: str, torch_version: str
) -> dict[str, str]:
    if (
        python != RUNTIME_PYTHON
        or transformers_version != TRANSFORMERS_VERSION
        or torch_version != TORCH_VERSION
    ):
        raise RuntimeError(
            "Runtime interpreter or package version differs from Base control"
        )
    return {
        "python": python,
        "transformers": transformers_version,
        "torch": torch_version,
    }


def _runtime() -> dict[str, str]:
    import torch
    import transformers

    return _validate_runtime(
        sys.executable, transformers.__version__, torch.__version__
    )


def _require_no_gpu() -> None:
    import torch

    if torch.cuda.device_count() != 0:
        raise RuntimeError("CPU preflight must not expose any GPU")


def _cpu_full_weight_source_load(source: Path, select: Path) -> dict[str, Any]:
    """Prove the exact native model loader works before any GPU allocation."""
    _require_no_gpu()
    from training.model.decision_model import DecisionModel, encode

    model, tokenizer = DecisionModel.from_base(
        source,
        POSTTRAINED_REVISION,
        head_dim=256,
        source_stage="posttrained",
    )
    if (
        model.metadata.get("backbone_model_type") != "qwen3_5"
        or model.metadata.get("source_stage") != "posttrained"
        or model.metadata.get("base_revision") != POSTTRAINED_REVISION
        or any(parameter.device.type != "cpu" for parameter in model.parameters())
    ):
        raise ValueError("CPU full-weight loader changed source or placement")
    rows = load_partition(select, "select")
    probes = {}
    for task_type in TYPES:
        row = next(row for row in rows if row["task_type"] == task_type)
        item = encode(row, tokenizer, 8192)
        probes[task_type] = {
            "token_count": len(item["ids"]),
            "prompt_sha256": item["prompt_sha256"],
            "token_ids_sha256": item["token_ids_sha256"],
        }
    result = {
        "source_revision": POSTTRAINED_REVISION,
        "text_parameter_count": model.metadata["text_parameter_count"],
        "head_dim": model.metadata["head_dim"],
        "native_type_probes": probes,
        "visible_gpu_count": 0,
    }
    del model, tokenizer
    gc.collect()
    return result


def _revision(source: Path) -> str:
    metadata_root = source / ".cache/huggingface/download"
    records = {
        path.read_text(encoding="utf-8").splitlines()[0]
        for path in metadata_root.rglob("*.metadata")
        if path.read_text(encoding="utf-8").splitlines()
    }
    if records != {POSTTRAINED_REVISION}:
        raise ValueError("Official source cache metadata is not the pinned revision")
    return POSTTRAINED_REVISION


def _source(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if _revision(source) != POSTTRAINED_REVISION:
        raise ValueError("Source revision differs")
    inventory = source_fingerprint(source)
    if any(inventory["files_sha256"].get(k) != v for k, v in SHARD_SHA256.items()):
        raise ValueError("Full source weight shards differ")
    if inventory["files_sha256"].get("config.json") != (
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
    ) or inventory["files_sha256"].get("tokenizer.json") != (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ):
        raise ValueError("Official source config or tokenizer differs")
    return inventory


def _data(paths: dict[str, Path]) -> dict[str, Any]:
    rows = {}
    for role, path in paths.items():
        if sha256_file(path) != DATA_SHA256[role]:
            raise ValueError(f"{role} bytes differ from frozen rights-clean v2")
        rows[role] = load_partition(path, role)
        if len(rows[role]) != COUNT[role]:
            raise ValueError(f"{role} row count differs")
    check_partition_isolation(rows)
    return {role: DATA_SHA256[role] for role in paths}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _runtime()
    _require_no_gpu()
    output = _private_root(args.output_root)
    if any(output.iterdir()):
        raise FileExistsError("New admission output directory must be empty")
    paths = {role: getattr(args, role).resolve(strict=True) for role in COUNT}
    source = args.source.resolve(strict=True)
    audit_path = args.cpu_audit.resolve(strict=True)
    if sha256_file(audit_path) != CPU_AUDIT_SHA256:
        raise ValueError("CPU-only source/input audit differs")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("posttrained_revision") != POSTTRAINED_REVISION
        or audit.get("exact_input_parity") is not True
        or audit.get("source_weights", {}).get("posttrained", {}).get("shard_sha256")
        != SHARD_SHA256
        or any(
            audit.get("panels", {}).get(role, {}).get("rows") != n
            for role, n in COUNT.items()
        )
        or audit.get("panels", {}).get("train", {}).get("posttrained_tokens")
        != 4_194_465
    ):
        raise ValueError("CPU audit did not admit exact source/input parity")
    if args.image_id != IMAGE_ID:
        raise ValueError("Training image ID differs from completed Base control")
    source_fingerprint = _source(source)
    data_sha256 = _data(paths)
    full_load = _cpu_full_weight_source_load(source, paths["select"])
    cpu_receipt = {
        "schema_version": SCHEMA,
        "phase": "cpu-full-weight-source-load",
        "image_id": IMAGE_ID,
        "runtime": runtime,
        "source_fingerprint": source_fingerprint,
        "data_sha256": data_sha256,
        "full_load": full_load,
        "code_sha256": _code_hashes(),
        "status": "PASS",
    }
    _write_once(output / "cpu-runtime-preflight.json", cpu_receipt)
    result = {
        "schema_version": SCHEMA,
        "status": "LOCKED_NO_GPU",
        "source_id": SOURCE_ID,
        "source_revision": POSTTRAINED_REVISION,
        "source_path": str(source),
        "source_fingerprint": source_fingerprint,
        "data_paths": {role: str(path) for role, path in paths.items()},
        "data_sha256": data_sha256,
        "cpu_audit_path": str(audit_path),
        "cpu_audit_sha256": CPU_AUDIT_SHA256,
        "image_id": IMAGE_ID,
        "code_sha256": _code_hashes(),
        "runtime": runtime,
        "cpu_runtime_preflight_sha256": sha256_file(
            output / "cpu-runtime-preflight.json"
        ),
        "output_root": str(output),
        "protocol": PROTOCOL,
    }
    _write_once(output / "admission-lock.json", result)
    return result


def _lock(path: Path) -> dict[str, Any]:
    runtime = _runtime()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Lock must be a regular private file")
    root = _private_root(path.parent)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("Lock must have mode 0600")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != SCHEMA
        or lock.get("status") != "LOCKED_NO_GPU"
        or lock.get("source_id") != SOURCE_ID
        or lock.get("source_revision") != POSTTRAINED_REVISION
        or lock.get("image_id") != IMAGE_ID
        or lock.get("output_root") != str(root)
        or lock.get("cpu_audit_sha256") != CPU_AUDIT_SHA256
        or lock.get("code_sha256") != _code_hashes()
        or lock.get("runtime") != runtime
        or lock.get("protocol") != PROTOCOL
    ):
        raise ValueError("Admission lock or code changed")
    if sha256_file(Path(lock["cpu_audit_path"])) != CPU_AUDIT_SHA256:
        raise ValueError("CPU audit report changed")
    cpu_preflight = root / "cpu-runtime-preflight.json"
    if sha256_file(cpu_preflight) != lock.get("cpu_runtime_preflight_sha256"):
        raise ValueError("CPU full-weight runtime preflight changed after lock")
    report = json.loads(cpu_preflight.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("runtime") != runtime
        or report.get("source_fingerprint") != lock.get("source_fingerprint")
        or report.get("data_sha256") != lock.get("data_sha256")
        or report.get("code_sha256") != lock.get("code_sha256")
        or report.get("full_load", {}).get("visible_gpu_count") != 0
    ):
        raise ValueError("CPU full-weight preflight did not match lock")
    if _source(Path(lock["source_path"])) != lock["source_fingerprint"]:
        raise ValueError("Source files changed after lock")
    if (
        _data({k: Path(v) for k, v in lock["data_paths"].items()})
        != lock["data_sha256"]
    ):
        raise ValueError("Data changed after lock")
    return lock


def _train_args(lock: dict[str, Any], output: Path, *, zero: bool) -> list[str]:
    data = lock["data_paths"]
    command = [
        RUNTIME_PYTHON,
        "-m",
        "training.model.train",
        "--model-path",
        lock["source_path"],
        "--init-kind",
        "posttrained",
        "--base-revision",
        POSTTRAINED_REVISION,
        "--train",
        data["train"],
        "--select",
        data["select"],
        "--cal",
        data["cal"],
        "--output",
        str(output),
        "--epochs",
        "1",
        "--max-steps",
        "466" if zero else "1",
        "--microbatch",
        "1",
        "--accumulation",
        "16",
        "--eval-batch",
        "2",
        "--max-length",
        "8192",
        "--head-dim",
        "256",
        "--train-mode",
        "lora",
        "--lora-rank",
        "16",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.05",
        "--lora-lr",
        "0.0001",
        "--head-lr",
        "0.0002",
        "--weight-decay",
        "0.01",
        "--warmup-ratio",
        "0.05",
        "--objective",
        "ce_brier",
        "--brier-weight",
        "0.5",
        "--save-every",
        "1",
        "--seed",
        "20260926",
    ]
    if zero:
        command.append("--zero-step-only")
    return command


def _predictions(path: Path, n: int) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != n or len({row.get("id") for row in records}) != n:
        raise ValueError("Prediction roster is incomplete or duplicated")
    return records


def _probs(row: dict[str, Any]) -> dict[str, float]:
    kind = row.get("task_type")
    answer = row.get("answer", {})
    if answer.get("type") != kind:
        raise ValueError("Native answer type differs from task type")
    if kind == "noul":
        p = answer.get("noul")
        values = {"false": 1 - p, "true": p} if type(p) in (int, float) else {}
    elif kind in {"choice", "score"}:
        values = answer.get("probabilities", {})
    else:
        values = {}
    if (
        not isinstance(values, dict)
        or len(values) < 2
        or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in values.values()
        )
        or abs(sum(values.values()) - 1) > 0.01
        or row.get("prediction_key") not in values
    ):
        raise ValueError("Invalid native probability answer")
    return values


def compare(first: Path, second: Path, n: int) -> dict[str, Any]:
    left, right = _predictions(first, n), _predictions(second, n)
    drift = []
    changes = 0
    types = {}
    for a, b in zip(left, right, strict=True):
        if any(
            a.get(k) != b.get(k)
            for k in ("id", "task_type", "prompt_sha256", "token_ids_sha256")
        ):
            raise ValueError("Native row identity or order changed")
        types[a["task_type"]] = types.get(a["task_type"], 0) + 1
        pa, pb = _probs(a), _probs(b)
        if pa.keys() != pb.keys():
            raise ValueError("Native option roster changed")
        changes += a["prediction_key"] != b["prediction_key"]
        drift.append(max(abs(pa[k] - pb[k]) for k in pa))
    maximum = max(drift)
    p99 = sorted(drift)[math.ceil(0.99 * len(drift)) - 1]
    passed = changes == 0 and p99 <= MAX_P99_DRIFT and maximum <= MAX_DRIFT
    return {
        "items": n,
        "types": types,
        "categorical_changes": changes,
        "probability_p99_drift": p99,
        "probability_max_drift": maximum,
        "status": "PASS" if passed else "FAIL",
    }


def _require_gpu() -> None:
    _runtime()
    import torch

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16 GPU must be visible")


def _run_train(lock_path: Path, phase: str) -> dict[str, Any]:
    start = time.monotonic()
    lock = _lock(lock_path)
    if phase not in {"zero-a", "zero-b", "one"}:
        raise ValueError("Unsupported GPU admission phase")
    if phase == "zero-b":
        first = Path(lock["output_root"]) / "zero-a-receipt.json"
        if not first.is_file() or json.loads(first.read_text(encoding="utf-8")).get(
            "lock_sha256"
        ) != sha256_file(lock_path):
            raise ValueError("First zero-step source start is missing or unbound")
    if phase == "one":
        _zero_gate(lock_path)
    _require_gpu()
    output = Path(lock["output_root"]) / phase
    if output.exists():
        raise FileExistsError("Admission phase cannot be rerun or overwritten")
    log_path = Path(lock["output_root"]) / f"{phase}-console.log"
    cap = ZERO_SECONDS if phase.startswith("zero") else ONE_SECONDS
    remaining = cap - (time.monotonic() - start)
    if remaining <= 0:
        raise TimeoutError("Admission preflight exceeded frozen phase cap")
    command = _train_args(lock, output, zero=phase.startswith("zero"))
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as log:
        try:
            subprocess.run(
                command,
                cwd=_repo_root(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            _write_once(
                Path(lock["output_root"]) / f"{phase}-failure.json",
                {
                    "schema_version": SCHEMA,
                    "phase": phase,
                    "lock_sha256": sha256_file(lock_path),
                    "elapsed_seconds": time.monotonic() - start,
                    "error_type": type(error).__name__,
                },
            )
            raise
    elapsed = time.monotonic() - start
    if elapsed > cap:
        raise TimeoutError("Admission phase exceeded frozen cap")
    baseline = output / "select-baseline-predictions.jsonl"
    records = _predictions(baseline, 700)
    expected = load_partition(lock["data_paths"]["select"], "select")
    if any(
        row.get("id") != source["id"] or row.get("task_type") != source["task_type"]
        for row, source in zip(records, expected, strict=True)
    ):
        raise ValueError("Admission changed SELECT row identity or order")
    if {
        kind: sum(row["task_type"] == kind for row in records) for kind in TYPES
    } != TYPES:
        raise ValueError("Admission omitted a native type")
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("model_source") != lock["source_fingerprint"]:
        raise ValueError("Trainer did not use the frozen source")
    if phase == "one":
        complete = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
        if (
            complete.get("status") != "complete"
            or complete.get("step") != 1
            or complete.get("planned_updates") != 1
        ):
            raise ValueError("One-update smoke did not complete exactly one step")
        events = [
            json.loads(line)
            for line in (output / "train-metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        updates = [event for event in events if event.get("event") == "train"]
        if (
            len(updates) != 1
            or updates[0].get("step") != 1
            or type(updates[0].get("tokens")) is not int
            or updates[0]["tokens"] < 1
            or any(
                not math.isfinite(updates[0].get(key, float("nan")))
                for key in ("loss", "gradient_norm")
            )
        ):
            raise ValueError(
                "One-update smoke has nonfinite or missing optimization evidence"
            )
        _predictions(output / "select-step-0000001-predictions.jsonl", 700)
    receipt = {
        "schema_version": SCHEMA,
        "phase": phase,
        "lock_sha256": sha256_file(lock_path),
        "image_id": lock["image_id"],
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "elapsed_seconds": elapsed,
        "baseline_sha256": sha256_file(baseline),
        "provenance_sha256": sha256_file(output / "provenance.json"),
        "status": "PASS",
    }
    if phase == "one":
        receipt["complete_sha256"] = sha256_file(output / "COMPLETE.json")
        receipt["step1_predictions_sha256"] = sha256_file(
            output / "select-step-0000001-predictions.jsonl"
        )
        receipt["optimizer_event"] = {
            key: updates[0][key] for key in ("step", "loss", "gradient_norm", "tokens")
        }
    _write_once(Path(lock["output_root"]) / f"{phase}-receipt.json", receipt)
    return receipt


def _zero_gate(lock_path: Path) -> dict[str, Any]:
    lock = _lock(lock_path)
    root = Path(lock["output_root"])
    for phase in ("zero-a", "zero-b"):
        receipt = json.loads(
            (root / f"{phase}-receipt.json").read_text(encoding="utf-8")
        )
        if (
            receipt.get("lock_sha256") != sha256_file(lock_path)
            or receipt.get("status") != "PASS"
        ):
            raise ValueError("Zero-step source start differs from lock")
        if sha256_file(
            root / phase / "select-baseline-predictions.jsonl"
        ) != receipt.get("baseline_sha256"):
            raise ValueError("Zero-step predictions changed after receipt")
    result = compare(
        root / "zero-a/select-baseline-predictions.jsonl",
        root / "zero-b/select-baseline-predictions.jsonl",
        700,
    )
    if result["types"] != TYPES:
        raise ValueError("Zero-step native type counts changed")
    result.update(
        {
            "schema_version": SCHEMA,
            "phase": "zero-gate",
            "lock_sha256": sha256_file(lock_path),
        }
    )
    gate_path = root / "zero-gate.json"
    if gate_path.exists():
        if json.loads(gate_path.read_text(encoding="utf-8")) != result:
            raise ValueError("Zero-step gate differs from the current predictions")
    else:
        _write_once(gate_path, result)
    if result["status"] != "PASS":
        raise ValueError("Zero-step repeatability failed")
    return result


def _reload(lock_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    lock = _lock(lock_path)
    _zero_gate(lock_path)
    root = Path(lock["output_root"])
    one = json.loads((root / "one-receipt.json").read_text(encoding="utf-8"))
    if one.get("lock_sha256") != sha256_file(lock_path) or one.get("status") != "PASS":
        raise ValueError("One-update receipt differs from lock")
    if sha256_file(root / "one/COMPLETE.json") != one.get(
        "complete_sha256"
    ) or sha256_file(root / "one/select-step-0000001-predictions.jsonl") != one.get(
        "step1_predictions_sha256"
    ):
        raise ValueError("One-update artifacts changed after receipt")
    checkpoint = root / "one/checkpoint-0000001"
    if not checkpoint.is_dir() or not (checkpoint / "checkpoint.json").is_file():
        raise ValueError("One-update checkpoint is missing")
    _require_gpu()
    import torch
    from training.model.data import load_partition
    from training.model.decision_model import DecisionModel, encode
    from training.model.train import evaluate

    def _timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Reload exceeded frozen cap")

    remaining = RELOAD_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("Reload preflight exceeded frozen cap")
    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(math.ceil(remaining))
    try:
        model, tokenizer = DecisionModel.from_checkpoint(
            checkpoint, source_path=lock["source_path"]
        )
        model = model.float().to(torch.device("cuda:0"))
        model.backbone.config.use_cache = False
        rows = load_partition(lock["data_paths"]["select"], "select")[:32]
        encoded = [encode(row, tokenizer, 8192) for row in rows]
        output = root / "one-reload"
        output.mkdir(mode=0o700)
        evaluate(
            model,
            encoded,
            pad_id=tokenizer.pad_token_id,
            batch_size=2,
            device=torch.device("cuda:0"),
            output=output,
            tag="reload32",
        )
        torch.cuda.synchronize()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    elapsed = time.monotonic() - started
    if elapsed > RELOAD_SECONDS:
        raise TimeoutError("Reload exceeded frozen cap")
    original = _predictions(root / "one/select-step-0000001-predictions.jsonl", 700)[
        :32
    ]
    original_path = output / "step1-first32.jsonl"
    descriptor = os.open(original_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in original:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    check = compare(original_path, output / "reload32-predictions.jsonl", 32)
    result = {
        "schema_version": SCHEMA,
        "phase": "reload",
        "lock_sha256": sha256_file(lock_path),
        "elapsed_seconds": elapsed,
        "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
        "predictions_sha256": sha256_file(output / "reload32-predictions.jsonl"),
        **check,
    }
    _write_once(root / "reload-receipt.json", result)
    if check["status"] != "PASS":
        raise ValueError("One-update native reload parity failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    prep = commands.add_parser("prepare")
    for name in ("source", "train", "select", "cal", "cpu-audit", "output-root"):
        prep.add_argument("--" + name, required=True, type=Path)
    prep.add_argument("--image-id", required=True)
    for name in ("verify", "zero-a", "zero-b", "compare-zero", "one", "reload"):
        phase = commands.add_parser(name)
        phase.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "lock_sha256": sha256_file(
                        args.output_root / "admission-lock.json"
                    ),
                }
            )
        )
    elif args.phase == "verify":
        _require_no_gpu()
        _lock(args.lock)
        print(
            json.dumps(
                {"status": "CPU_LOCK_VERIFIED", "lock_sha256": sha256_file(args.lock)}
            )
        )
    elif args.phase == "compare-zero":
        _require_no_gpu()
        result = _zero_gate(args.lock)
        print(json.dumps(result, sort_keys=True))
    elif args.phase == "reload":
        result = _reload(args.lock)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
        )
    else:
        result = _run_train(args.lock, args.phase)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
        )


if __name__ == "__main__":
    main()
