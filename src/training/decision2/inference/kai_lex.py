"""Collect published Kai/Lex 1.0 predictions through native SystemOne.

Each process loads one verified release. The published inference profile uses
FP32 on one visible AMD ROCm GPU and rejects complete inputs over 1024 tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from inference.run import digest, file_digest, load_prompts, local_revision, synchronize

ADAPTER_VERSION = "kai-lex-native-v1"
MODELS = {
    "kai": {
        "model_id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
        "model_name": "Decision-1.0-Kai",
        "revision": "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
        "manifest_sha256": "c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4",
    },
    "lex": {
        "model_id": "llm-semantic-router/Decision-1.0-Lex-0.6B",
        "model_name": "Decision-1.0-Lex",
        "revision": "ee8e74d912fca8328a353c11d174b44da3f91781",
        "manifest_sha256": "f288d873999832a3f37c6a7c4268c2ab309691e621794dbf7acab891acbbb7e6",
    },
}
QUALIFIED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.12.0+git6bbd260",
    "hip": "7.2.53211",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
    "numpy": "2.5.3",
    "device_architecture": "gfx942",
}
OVERFLOW = re.compile(r"\bexceeds 1024 tokens; no implicit truncation\b")


def verify_native_bundle(path: Path, backend: str, revision: str) -> dict[str, Any]:
    """Check HF revision metadata and every byte in the native file roster."""
    expected = MODELS[backend]
    if revision != expected["revision"]:
        raise ValueError(
            f"{backend} requires published revision {expected['revision']}"
        )
    if not local_revision(path, revision):
        raise ValueError(
            "HF local-dir metadata is required to attest the exact revision"
        )
    native = path / "native"
    manifest_path = native / "MANIFEST.json"
    if file_digest(manifest_path) != expected["manifest_sha256"]:
        raise ValueError("Native manifest SHA-256 differs from the pinned release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("schema") != "decision.files.v1" or not isinstance(files, dict):
        raise ValueError("Unexpected native manifest schema")
    for name, reference in files.items():
        if not isinstance(name, str) or not isinstance(reference, dict):
            raise ValueError("Malformed native manifest entry")
        rel = PurePosixPath(name)
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or rel.as_posix() != name
            or "\\" in name
            or set(reference) != {"bytes", "sha256"}
        ):
            raise ValueError("Unsafe native manifest entry")
        item = native / name
        if (
            not item.is_file()
            or item.is_symlink()
            or item.stat().st_size != reference["bytes"]
            or file_digest(item) != reference["sha256"]
        ):
            raise ValueError(f"Native bundle integrity check failed: {name}")
    present = set()
    for item in native.rglob("*"):
        if item.is_symlink():
            raise ValueError("Native bundle contains a symlink")
        rel = item.relative_to(native)
        if item.is_file() and not (
            "__pycache__" in rel.parts and item.suffix == ".pyc"
        ):
            present.add(rel.as_posix())
    if present != set(files) | {"MANIFEST.json"}:
        raise ValueError("Native file roster differs from the pinned manifest")
    config = json.loads((native / "decision_config.json").read_text(encoding="utf-8"))
    required = {
        "architecture": "vela_decision_score_path_capacity_v1",
        "schema": "decision.nano.score_path_capacity.v1",
        "arm": "all22",
        "training_arm": "S22",
        "base_model": "llm-semantic-router/Vela-1.0-Encoder-307M",
        "transformers_version": "4.57.6",
        "weight_dtype": "float32",
        "inference_precision": "fp32",
        "calibration": "none; raw probabilities",
    }
    mismatch = {
        key: (value, config.get(key))
        for key, value in required.items()
        if config.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Unexpected native architecture/configuration: {mismatch}")
    return {
        "backend": backend,
        "model_id": expected["model_id"],
        "model_name": expected["model_name"],
        "model_revision": revision,
        "revision_attested": True,
        "model_config_sha256": expected["manifest_sha256"],
        "native_file_count": len(files),
    }


def runtime_report(device: str, *, check_gpu: bool) -> dict[str, Any]:
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers

    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
    }
    if check_gpu:
        if (
            device != "cuda:0"
            or torch.version.hip is None
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
        ):
            raise RuntimeError("Expose exactly one AMD ROCm GPU as cuda:0")
        actual["device_architecture"] = str(
            getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unknown")
        ).split(":", 1)[0]
    expected = {
        key: value
        for key, value in QUALIFIED_RUNTIME.items()
        if check_gpu or key != "device_architecture"
    }
    differences = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    return {
        "runtime_matches_validated": not differences,
        "runtime_differences": differences,
        "runtime_actual": actual,
    }


def load_system_one(path: Path, backend: str, device: str):
    """Import one verified package, then use its documented loader and API."""
    for name in ("decision_runtime", "decision_inference"):
        loaded = sys.modules.get(name)
        if loaded is not None:
            module_file = getattr(loaded, "__file__", None)
            if (
                module_file is None
                or Path(module_file).resolve().parent != (path / name).resolve()
            ):
                raise RuntimeError(
                    "A different Kai/Lex native package is loaded in this process"
                )
    sys.path.insert(0, str(path))
    import torch
    from decision_inference import SystemOne
    from decision_runtime import load_native

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    expected = MODELS[backend]
    native = load_native(
        path / "native",
        expected_manifest_sha256=expected["manifest_sha256"],
        device=device,
    )
    return native, SystemOne(native, model=expected["model_name"], batching="default")


def completed_ids(
    output: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: {
            "hash": digest({"state": row["state"], "questions": row["questions"]}),
            "question_ids": set(row["questions"]),
        }
        for row in rows
    }
    completed = set()
    with output.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            item = json.loads(line)
            item_id = item.get("id")
            if item_id not in expected or item_id in completed:
                raise ValueError(
                    f"{output}:{number}: unknown or duplicate prediction ID"
                )
            for key in (
                "backend",
                "model_id",
                "model_revision",
                "revision_attested",
                "model_config_sha256",
            ):
                if item.get(key) != identity[key]:
                    raise ValueError(f"{output}:{number}: stale model identity")
            if (
                item.get("adapter_version") != ADAPTER_VERSION
                or item.get("source_input_sha256") != expected[item_id]["hash"]
                or not isinstance(item.get("answers"), dict)
                or set(item["answers"]) != expected[item_id]["question_ids"]
            ):
                raise ValueError(f"{output}:{number}: stale input or adapter")
            completed.add(item_id)
    return completed


def collect(
    *,
    backend: str,
    model_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    resume: bool = False,
    allow_unvalidated_runtime: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if backend not in MODELS:
        raise ValueError(f"Unsupported backend: {backend}")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    rows = load_prompts(prompts)
    for row in rows:
        if any(not isinstance(q, dict) for q in row["questions"].values()):
            raise ValueError(f"{row['id']}: questions must be objects")
        kinds = {question.get("type") for question in row["questions"].values()}
        if kinds - {"choice", "noul", "score"}:
            raise ValueError(
                f"{row['id']}: unsupported question types {kinds - {'choice', 'noul', 'score'}}"
            )
    model_path = model_path.resolve(strict=True)
    identity = verify_native_bundle(model_path, backend, revision)
    runtime = runtime_report(device, check_gpu=True)
    if not runtime["runtime_matches_validated"] and not allow_unvalidated_runtime:
        raise RuntimeError(
            f"Runtime differs from the published validation: {runtime['runtime_differences']}"
        )
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"{output} exists; --resume requires the same input and model"
            )
        completed = completed_ids(output, rows, identity)
    else:
        completed = set()
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    if not remaining:
        return {
            **identity,
            **runtime,
            "input_items": len(rows),
            "previously_completed": len(completed),
            "collected_now": 0,
            "output": str(output),
        }
    native, client = load_system_one(model_path, backend, device)
    assert native is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    overflow_count = 0
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            try:
                response = client.system_one(**payload)
                invalid_reason = None
            except ValueError as exc:
                if not OVERFLOW.search(str(exc)):
                    raise
                invalid_reason = "context_overflow"
                overflow_count += 1
                response = {
                    "model": identity["model_name"],
                    "answers": {
                        qid: {"type": question["type"], "error": invalid_reason}
                        for qid, question in row["questions"].items()
                    },
                    "usage": None,
                }
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if (
                not isinstance(response, dict)
                or response.get("model") != identity["model_name"]
                or not isinstance(response.get("answers"), dict)
                or response["answers"].keys() != row["questions"].keys()
                or not math.isfinite(latency_ms)
                or latency_ms < 0
            ):
                raise ValueError(f"{row['id']}: malformed native SystemOne response")
            receipt = {
                "id": row["id"],
                "answers": response["answers"],
                "latency_ms": latency_ms,
                "usage": response.get("usage"),
                "model": response["model"],
                "backend": backend,
                "model_id": identity["model_id"],
                "adapter_version": ADAPTER_VERSION,
                "model_revision": revision,
                "revision_attested": True,
                "model_config_sha256": identity["model_config_sha256"],
                "source_input_sha256": digest(payload),
                "invalid_reason": invalid_reason,
                "runtime_matches_validated": runtime["runtime_matches_validated"],
                "runtime_differences": runtime["runtime_differences"],
            }
            target.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            target.flush()
    return {
        **identity,
        **runtime,
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "context_overflow_now": overflow_count,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(MODELS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path, help="Gold-free benchmark prompts JSONL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unvalidated-runtime", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--verify-only", action="store_true", help="CPU-only bundle/runtime preflight"
    )
    args = parser.parse_args()
    if args.verify_only:
        identity = verify_native_bundle(
            args.model_path.resolve(strict=True), args.backend, args.model_revision
        )
        print(
            json.dumps(
                {**identity, **runtime_report(args.device, check_gpu=False)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --verify-only is set")
    report = collect(
        backend=args.backend,
        model_path=args.model_path,
        revision=args.model_revision,
        prompts=args.input,
        output=args.output,
        device=args.device,
        resume=args.resume,
        allow_unvalidated_runtime=args.allow_unvalidated_runtime,
        max_items=args.max_items,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
