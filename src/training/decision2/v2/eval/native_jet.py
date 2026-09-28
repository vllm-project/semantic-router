"""Collect Jet v6.2's native answers (bundled ``jet.Jet().decide``) on gold-free prompts.

The release ships top-level modules named ``inference``, ``format`` and ``runtime``,
which would collide with this repository's ``inference`` package, so this collector
lives outside it and imports nothing from it. Every release file is checked against
``release-manifest.json`` before loading. Prompts over the native 16,384-token limit
are rejected by the release without truncation and recorded as invalid answers. The
release targets Linux + NVIDIA CUDA; ROCm runs are labelled ``unvalidated_rocm``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from v2.eval.same_panel import input_digest, sha_file

MODEL_ID = "michaljach/jet"
MODEL_REVISION = "fbc3d2daa679e0d4bd9f99c9912b6496d5a41f0a"
ADAPTER_VERSION = "jet-v6.2-native-v1"


def attested_revision(model_path: Path) -> str | None:
    revisions = {
        (p.read_text(encoding="utf-8").splitlines() or [""])[0].strip()
        for p in (model_path / ".cache/huggingface/download").rglob("*.metadata")
    }
    return next(iter(revisions)) if len(revisions) == 1 else None


def verify_release(model_path: Path, revision: str) -> dict[str, Any]:
    if revision != MODEL_REVISION or attested_revision(model_path) != revision:
        raise ValueError("Jet requires its attested, pinned Hugging Face revision")
    manifest = json.loads(
        (model_path / "release-manifest.json").read_text(encoding="utf-8")
    )
    files = manifest.get("files")
    if manifest.get("version") != "v6.2.0" or not isinstance(files, dict) or not files:
        raise ValueError("Unexpected Jet release manifest")
    for name, expected in files.items():
        if sha_file(model_path / name) != expected:
            raise ValueError(f"Jet release file differs from its manifest: {name}")
    for name in (
        "jet.py",
        "inference.py",
        "format.py",
        "runtime.py",
        "calibration.json",
    ):
        if not (model_path / name).is_file():
            raise ValueError(f"Jet release lacks {name}")
    return {
        "manifest_sha256": sha_file(model_path / "release-manifest.json"),
        "verified_files": len(files),
    }


def invalid_answers(questions: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        name: {"type": q.get("type"), "invalid_reason": reason}
        for name, q in questions.items()
    }


def collect(
    *, model_path: Path, revision: str, prompts: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_path = model_path.resolve(strict=True)
    identity = verify_release(model_path, revision)
    sys.path.insert(0, str(model_path))
    import torch
    from jet import Jet

    engine = Jet(str(model_path))
    rejected = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            torch.cuda.synchronize()
            started = time.perf_counter()
            error = None
            try:
                answers = engine.decide(row["state"], row["questions"])["answers"]
            except ValueError as exc:
                answers, error = (
                    invalid_answers(row["questions"], "native_rejection"),
                    str(exc)[:200],
                )
                rejected += 1
            torch.cuda.synchronize()
            if answers.keys() != row["questions"].keys():
                raise ValueError(f"{row['id']}: Jet answers do not match question IDs")
            target.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": answers,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "native_error": error,
                        "source_input_sha256": input_digest(
                            row["state"], row["questions"]
                        ),
                        "backend": "jet-native",
                        "model_id": MODEL_ID,
                        "model_revision": revision,
                        "adapter_version": ADAPTER_VERSION,
                        "model_config_sha256": identity["manifest_sha256"],
                        "runtime_qualification": "unvalidated_rocm",
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            target.flush()
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "items": len(rows),
        "native_rejections": rejected,
        **identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                revision=args.model_revision,
                prompts=args.input,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
