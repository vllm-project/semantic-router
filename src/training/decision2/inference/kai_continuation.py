"""Collect Kai-derived Decision 2.0 predictions from an exact native export.

This adapter uses the published Kai SystemOne wrapper and its native loader.
The completed fine-tune receipt, exported manifest, file roster and qualified
runtime are checked before any gold-free input is sent to the model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from inference.kai_lex import MODELS, runtime_report, verify_native_bundle
from inference.run import digest, file_digest, load_prompts, synchronize

VERSION = "kai-native-continuation-v1"
MODEL_ID = "llm-semantic-router/dev-2.0-0.6b"
MODEL_NAME = "dev-2.0-0.6b"
PARENT_MANIFEST_SHA = "c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4"


def verify_export(run: Path, native: Path, expected_manifest: str) -> dict[str, Any]:
    """Require a completed run and every original export file to match bytes."""
    if len(expected_manifest) != 64 or any(
        c not in "0123456789abcdef" for c in expected_manifest
    ):
        raise ValueError("Expected native manifest must be a SHA-256 digest")
    if (
        run.is_symlink()
        or native.is_symlink()
        or not run.is_dir()
        or not native.is_dir()
        or not native.is_relative_to(run)
    ):
        raise ValueError(
            "Native export must be a real subdirectory of its completed run"
        )
    complete_path = run / "COMPLETE.json"
    if complete_path.is_symlink() or not complete_path.is_file():
        raise ValueError("Native export lacks a completed run receipt")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest_path = native / "MANIFEST.json"
    if manifest_path.is_symlink() or file_digest(manifest_path) != expected_manifest:
        raise ValueError("Native export manifest SHA-256 differs")
    if (
        complete.get("status") != "COMPLETE_DECISION_FINETUNE"
        or complete.get("native") != str(native)
        or complete.get("native_manifest_sha256") != expected_manifest
        or complete.get("identity", {}).get("native_manifest_sha256")
        != PARENT_MANIFEST_SHA
    ):
        raise ValueError(
            "Native export differs from completed training and parent source"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if (
        manifest.get("schema") != "decision.files.v1"
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError("Unknown native file manifest")
    seen = set()
    for name, receipt in files.items():
        rel = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or rel.is_absolute()
            or ".." in rel.parts
            or rel.as_posix() != name
            or "\\" in name
            or not isinstance(receipt, dict)
            or set(receipt) != {"bytes", "sha256"}
        ):
            raise ValueError("Malformed native export roster")
        path = native / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or file_digest(path) != receipt["sha256"]
        ):
            raise ValueError(f"Native export file differs: {name}")
        seen.add(name)
    actual = set()
    for path in native.rglob("*"):
        if path.is_symlink():
            raise ValueError("Native export contains a symlink")
        if path.is_file() and "__pycache__" not in path.parts:
            actual.add(path.relative_to(native).as_posix())
    if actual != seen | {"MANIFEST.json"}:
        raise ValueError("Native export contains extra or missing files")
    return {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "model_revision": expected_manifest,
        "model_config_sha256": expected_manifest,
        "base_manifest_sha256": PARENT_MANIFEST_SHA,
        "run_receipt_sha256": file_digest(complete_path),
        "native_file_count": len(files),
        "revision_attested": True,
    }


def _completed(
    output: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in rows
    }
    completed = set()
    for number, line in enumerate(output.open(encoding="utf-8"), 1):
        item = json.loads(line)
        item_id = item.get("id")
        if item_id not in expected or item_id in completed:
            raise ValueError(f"Stale or duplicate prediction at line {number}")
        if (
            item.get("model_id") != identity["model_id"]
            or item.get("model_revision") != identity["model_revision"]
            or item.get("model_config_sha256") != identity["model_config_sha256"]
            or item.get("base_manifest_sha256") != identity["base_manifest_sha256"]
            or item.get("run_receipt_sha256") != identity["run_receipt_sha256"]
            or item.get("adapter_version") != VERSION
            or item.get("source_input_sha256") != expected[item_id][0]
            or not isinstance(item.get("answers"), dict)
            or set(item["answers"]) != expected[item_id][1]
        ):
            raise ValueError(f"Stale model/input identity at line {number}")
        completed.add(item_id)
    return completed


def collect(
    *,
    run: Path,
    native: Path,
    base: Path,
    expected_manifest: str,
    prompts: Path,
    output: Path,
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    run, native = run.resolve(strict=True), native.resolve(strict=True)
    rows = load_prompts(prompts)
    identity = verify_export(run, native, expected_manifest)
    runtime = runtime_report("cuda:0", check_gpu=True)
    if not runtime["runtime_matches_validated"]:
        raise RuntimeError(
            f"Native Kai runtime differs: {runtime['runtime_differences']}"
        )
    if output.exists() and not resume:
        raise FileExistsError(output)
    completed = _completed(output, rows, identity) if output.exists() else set()
    remaining = [row for row in rows if row["id"] not in completed]
    if max_items is not None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
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
    base = base.resolve(strict=True)
    verify_native_bundle(base, "kai", MODELS["kai"]["revision"])
    sys.path.insert(0, str(base))
    import torch
    from decision_inference import SystemOne
    from decision_runtime import load_native

    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    model = load_native(
        native, expected_manifest_sha256=expected_manifest, device="cuda:0"
    )
    client = SystemOne(model, model=MODEL_NAME, batching="default")
    output.parent.mkdir(parents=True, exist_ok=True)
    overflows = 0
    with output.open("a" if output.exists() else "x", encoding="utf-8") as stream:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize("cuda:0")
            start = time.perf_counter()
            try:
                response = client.system_one(**payload)
                invalid = None
            except ValueError as error:
                if "exceeds 1024 tokens; no implicit truncation" not in str(error):
                    raise
                invalid = "context_overflow"
                overflows += 1
                response = {
                    "model": MODEL_NAME,
                    "usage": None,
                    "answers": {
                        qid: {"type": question["type"], "error": invalid}
                        for qid, question in row["questions"].items()
                    },
                }
            synchronize("cuda:0")
            latency = (time.perf_counter() - start) * 1000
            if (
                not isinstance(response, dict)
                or response.get("model") != MODEL_NAME
                or not isinstance(response.get("answers"), dict)
                or set(response["answers"]) != set(row["questions"])
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError("Malformed native SystemOne result")
            receipt = {
                "id": row["id"],
                "answers": response["answers"],
                "latency_ms": latency,
                "usage": response.get("usage"),
                "model": MODEL_NAME,
                "backend": "kai-continuation",
                "model_id": MODEL_ID,
                "model_revision": expected_manifest,
                "model_config_sha256": expected_manifest,
                "base_manifest_sha256": PARENT_MANIFEST_SHA,
                "run_receipt_sha256": identity["run_receipt_sha256"],
                "adapter_version": VERSION,
                "revision_attested": True,
                "source_input_sha256": digest(payload),
                "invalid_reason": invalid,
                "runtime_matches_validated": True,
                "runtime_differences": {},
            }
            stream.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            stream.flush()
    return {
        **identity,
        **runtime,
        "input_items": len(rows),
        "previously_completed": len(completed),
        "collected_now": len(remaining),
        "context_overflow_now": overflows,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                run=args.run,
                native=args.native,
                base=args.base_model,
                expected_manifest=args.manifest_sha256,
                prompts=args.input,
                output=args.output,
                resume=args.resume,
                max_items=args.max_items,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
