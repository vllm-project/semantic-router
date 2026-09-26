"""Collect native Decision 1.0 or Decider predictions for gold-free JSONL prompts.

Load one published model in one process. The resulting JSONL uses the same
``id``/``answers``/``latency_ms`` contract as the official Jev collector.
Model load time is excluded from per-request latency. A partial output can be
continued with --resume after validating every existing input fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ADAPTER_VERSION = "native-published-v2"
DECISION_BACKENDS = {
    "lux": (
        "llm-semantic-router/Decision-1.0-Lux-9B",
        "Decision-1.0-Lux",
        "Qwen/Qwen3.5-9B",
    ),
    "nox": (
        "llm-semantic-router/Decision-1.0-Nox-4B",
        "Decision-1.0-Nox",
        "Qwen/Qwen3.5-4B",
    ),
    "sol": (
        "llm-semantic-router/Decision-1.0-Sol-2B",
        "Decision-1.0-Sol",
        "Qwen/Qwen3.5-2B",
    ),
    "eos": (
        "llm-semantic-router/Decision-1.0-Eos-0.8B",
        "Decision-1.0-Eos",
        "Qwen/Qwen3.5-0.8B",
    ),
}
BACKENDS = frozenset(DECISION_BACKENDS) | {"decider"}


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    content = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            content.update(block)
    return content.hexdigest()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: expected row with string id")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row['id']}")
            if "gold" in row:
                raise ValueError(
                    f"{path}:{line_number}: gold-bearing input is forbidden"
                )
            if not isinstance(row.get("questions"), dict) or not row["questions"]:
                raise ValueError(
                    f"{path}:{line_number}: expected nonempty questions map"
                )
            if "state" not in row:
                raise ValueError(f"{path}:{line_number}: state is missing")
            # Reject non-JSON scalar values and make the benchmark input digest
            # independent of the file's whitespace formatting.
            digest({"state": row["state"], "questions": row["questions"]})
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prompts")
    return rows


def completed_rows(
    path: Path,
    rows: list[dict[str, Any]],
    backend: str,
    revision: str,
    model_config_sha256: str,
    model_id: str | None = None,
    revision_attested: bool = False,
) -> set[str]:
    expected = {
        row["id"]: {
            "input_sha256": digest(
                {"state": row["state"], "questions": row["questions"]}
            ),
            "question_ids": row["questions"].keys(),
        }
        for row in rows
    }
    completed = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            item_id = item.get("id")
            if item_id not in expected or item_id in completed:
                raise ValueError(
                    f"{path}:{line_number}: unknown or duplicate id {item_id!r}"
                )
            if (
                item.get("source_input_sha256") != expected[item_id]["input_sha256"]
                or item.get("backend") != backend
                or item.get("adapter_version") != ADAPTER_VERSION
                or item.get("model_revision") != revision
                or item.get("model_config_sha256") != model_config_sha256
                or item.get("model_id") != model_id
                or item.get("revision_attested") != revision_attested
            ):
                raise ValueError(
                    f"{path}:{line_number}: stale input, model, or adapter"
                )
            if (
                not isinstance(item.get("answers"), dict)
                or item["answers"].keys() != expected[item_id]["question_ids"]
            ):
                raise ValueError(f"{path}:{line_number}: missing or mismatched answers")
            completed.add(item_id)
    return completed


def assert_native_package_path(package_root: Path) -> None:
    """Native releases use the same ``decision`` package name; isolate processes."""
    existing = sys.modules.get("decision")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if (
            module_file is None
            or Path(module_file).resolve().parent != package_root.resolve()
        ):
            raise RuntimeError(
                "A different Decision package is already imported; run one model per process"
            )


def local_revision(path: Path, revision: str) -> bool:
    """Cross-check HF local-dir metadata when present; never infer a revision."""
    metadata_root = path / ".cache/huggingface/download"
    if not metadata_root.is_dir():
        return False
    revisions = set()
    for file in metadata_root.rglob("*.metadata"):
        lines = file.read_text(encoding="utf-8").splitlines()
        if lines:
            revisions.add(lines[0].strip())
    if not revisions:
        return False
    if revisions != {revision}:
        raise ValueError(
            f"Downloaded model files do not attest requested revision {revision}"
        )
    return True


def verify_model_family(path: Path, backend: str) -> None:
    if backend not in DECISION_BACKENDS:
        return
    config = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    expected_base = DECISION_BACKENDS[backend][2]
    if config.get("base_model") != expected_base:
        raise ValueError(
            f"{backend} expects base_model={expected_base}; got {config.get('base_model')}"
        )
    native_name = config.get("model_name")
    if native_name is not None and native_name != DECISION_BACKENDS[backend][1]:
        raise ValueError(f"{backend} native model name mismatch: {native_name}")


def verify_eos_manifest(path: Path) -> None:
    manifest = json.loads((path / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("model_name") != DECISION_BACKENDS["eos"][1] or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("Invalid Eos model manifest")
    required = {
        "decision/engine.py",
        "decision/model.py",
        "decision/types.py",
        "decision_config.json",
        "decision_head.safetensors",
        "runtime.json",
        "tokenizer.json",
        "backbone/config.json",
    }
    files = manifest["files"]
    if not required <= set(files) or not any(
        name.startswith("backbone/") and name.endswith(".safetensors") for name in files
    ):
        raise ValueError("Eos model manifest omits required inference files")
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Eos model manifest has an unsafe path")
        file = path / relative
        if (
            not file.is_file()
            or file.stat().st_size != expected.get("bytes")
            or file_digest(file) != expected.get("sha256")
        ):
            raise ValueError(f"Eos bundle integrity check failed: {name}")


def eos_runtime_report(
    path: Path, device: str, allow_unvalidated_runtime: bool
) -> dict[str, Any]:
    if not device.startswith("cuda"):
        raise RuntimeError("Eos requires a CUDA/ROCm GPU device")
    try:
        import fla
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Eos requires the qualified PyTorch, Transformers, and FLA runtime"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Eos requires a CUDA/ROCm GPU device")
    expected = json.loads((path / "runtime.json").read_text(encoding="utf-8"))[
        "qualified_runtime"
    ]
    properties = torch.cuda.get_device_properties(device)
    architecture = str(getattr(properties, "gcnArchName", "unknown")).split(":", 1)[0]
    config = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    actual = {
        "torch": str(torch.__version__),
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
        "flash_linear_attention": fla.__version__,
        "device_architecture": architecture,
        "attention": config.get("attention"),
    }
    differences = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if differences and not allow_unvalidated_runtime:
        raise RuntimeError(
            f"Eos runtime differs from the qualified release: {differences}"
        )
    return {
        "runtime_matches_validated": not differences,
        "runtime_differences": differences,
        "temperature": json.loads((path / "runtime.json").read_text())["temperature"],
    }


def load_pointer_bundle(path: Path, device: str, allow_unvalidated_runtime: bool):
    if not (path / "src/decision/model.py").is_file():
        raise FileNotFoundError(f"Decision native bundle loader missing from {path}")
    assert_native_package_path(path / "src/decision")
    sys.path.insert(0, str(path / "src"))
    from decision.model import DecisionModel

    model = DecisionModel.from_pretrained(
        str(path),
        device=device,
        local_files_only=True,
        allow_unvalidated_runtime=allow_unvalidated_runtime,
    )
    return (
        model,
        model.decide,
        {
            "runtime_matches_validated": model.runtime["matches_validated_runtime"],
            "runtime_differences": model.runtime["differences"],
        },
    )


def load_eos(path: Path, device: str, allow_unvalidated_runtime: bool):
    if not (path / "decision/engine.py").is_file():
        raise FileNotFoundError(f"Eos native engine missing from {path}")
    verify_eos_manifest(path)
    runtime = eos_runtime_report(path, device, allow_unvalidated_runtime)
    assert_native_package_path(path / "decision")
    sys.path.insert(0, str(path))
    from decision import DecisionModel

    model = DecisionModel.from_pretrained(
        str(path), local_files_only=True, device=device
    )
    return model, model.decide, runtime


def load_decider(path: Path, device: str):
    if not (path / "decider/infer.py").is_file():
        raise FileNotFoundError(f"Decider native inference package missing from {path}")
    sys.path.insert(0, str(path))
    from decider.infer import Decider

    # Eager mode is the published model's native `system_one` path and does
    # not require CUDA graph support from the ROCm host.
    model = Decider(str(path), device=device, use_graphs=False)
    return (
        model,
        model.system_one,
        {
            "runtime_matches_validated": None,
            "runtime_differences": None,
            "temperature": model.T,
            "temperature_by_type": model.T_by_type,
            "isolated_score_levels": model.isolated_levels,
            "layout": model.layout,
        },
    )


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


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
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if not revision:
        raise ValueError("model revision must be a pinned commit SHA or immutable tag")
    if backend == "decider" and allow_unvalidated_runtime:
        raise ValueError("--allow-unvalidated-runtime only applies to Decision models")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    rows = load_prompts(prompts)
    model_path = model_path.resolve(strict=True)
    model_id = DECISION_BACKENDS[backend][0] if backend in DECISION_BACKENDS else None
    if backend in DECISION_BACKENDS:
        verify_model_family(model_path, backend)
    revision_attested = local_revision(model_path, revision)
    config_name = (
        "bundle-manifest.json"
        if backend in {"lux", "nox", "sol"}
        else "MODEL_MANIFEST.json" if backend == "eos" else "decider_config.json"
    )
    config_sha256 = file_digest(model_path / config_name)
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"{output} exists; use --resume only for this same run"
            )
        completed = completed_rows(
            output, rows, backend, revision, config_sha256, model_id, revision_attested
        )
    else:
        completed = set()
    if backend in {"lux", "nox", "sol"}:
        model, decide, runtime = load_pointer_bundle(
            model_path, device, allow_unvalidated_runtime
        )
    elif backend == "eos":
        model, decide, runtime = load_eos(model_path, device, allow_unvalidated_runtime)
    else:
        model, decide, runtime = load_decider(model_path, device)
    # Keep `model` alive for the duration of collection.
    assert model is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        remaining = remaining[:max_items]
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            response = decide(**payload)
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if not isinstance(response, dict) or not isinstance(
                response.get("answers"), dict
            ):
                raise ValueError(f"{row['id']}: native model returned no answer map")
            if (
                backend in DECISION_BACKENDS
                and response.get("model") != DECISION_BACKENDS[backend][1]
            ):
                raise ValueError(
                    f"{row['id']}: native model identity disagrees with {backend} backend"
                )
            if response["answers"].keys() != row["questions"].keys():
                raise ValueError(
                    f"{row['id']}: native answer IDs do not match question IDs"
                )
            if not math.isfinite(latency_ms) or latency_ms < 0:
                raise ValueError(f"{row['id']}: invalid latency")
            receipt = {
                "id": row["id"],
                "answers": response["answers"],
                "latency_ms": latency_ms,
                "usage": response.get("usage"),
                "model": response.get("model"),
                "backend": backend,
                "model_id": model_id,
                "adapter_version": ADAPTER_VERSION,
                "model_revision": revision,
                "revision_attested": revision_attested,
                "model_config_sha256": config_sha256,
                "source_input_sha256": digest(payload),
                **runtime,
            }
            target.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            target.flush()
    return {
        "backend": backend,
        "model_id": model_id,
        "revision": revision,
        "revision_attested": revision_attested,
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "output": str(output),
        "model_config_sha256": config_sha256,
        **runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(sorted(BACKENDS)), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="Gold-free benchmark prompts JSONL"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unvalidated-runtime", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    result = collect(
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
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
