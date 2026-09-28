"""Collect Hopper (G) 1.2's native answers (``hopper_decisions.Decider``) on gold-free prompts.

Hopper (G) is a LoRA adapter on Qwen/Qwen3.5-4B served by the Hopper code at tag
``g-1.2.0``. The adapter files are verified against the release ``CHECKSUMS.txt``, the
serving source must be the pinned commit, the base loads from the offline local cache,
and the adapter's calibration map must equal the one packaged with the serving code
(the release says it is served unchanged). ``Decider`` is used as ``hopper-serve`` builds
it, with its default long-menu shortlist, except ``allow_slow_kernels=True``: the fused
CUDA causal-conv1d kernel is unavailable on ROCm, so the linear-attention convolution
runs on the PyTorch fallback (speed only; disclosed, run as ``unvalidated_rocm``).

The server takes one question per request, so each question is sent alone, exactly as
the JevBench harness does. Its Score answer carries only level probabilities; the
scorer's ``score`` field is their probability-weighted mean, which is how the harness
reads it. Hopper renders the full state without truncation; any native error stops the
run. Research-and-demo licence: internal comparison only, never on cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from v2.eval.native_jet import attested_revision
from v2.eval.same_panel import input_digest, sha_file

MODEL_ID = "HopitAI/hopper-g"
MODEL_REVISION = "71d991f4df78c445a50a4bc952748cea89c2ac21"
SOURCE_COMMIT = "0204f92954f76fd8835f6050c271fc6e75ac3370"
ADAPTER_VERSION = "hopper-g-1.2-native-v1"


def verify_release(model_path: Path, revision: str, source: Path) -> dict[str, Any]:
    if revision != MODEL_REVISION or attested_revision(model_path) != revision:
        raise ValueError(
            "Hopper (G) requires its attested, pinned Hugging Face revision"
        )
    sums = {}
    for line in (model_path / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    if "adapter_model.safetensors" not in sums:
        raise ValueError("Hopper (G) CHECKSUMS.txt lacks the adapter weights")
    for name, expected in sums.items():
        if sha_file(model_path / name) != expected:
            raise ValueError(f"Hopper (G) file differs from CHECKSUMS.txt: {name}")
    head = (source / ".git/HEAD").read_text(encoding="utf-8").strip()
    if head != SOURCE_COMMIT:
        raise ValueError("Hopper serving source is not the pinned g-1.2.0 commit")
    packaged_map = source / "hopper_decisions/maps/hopper.json"
    if sha_file(packaged_map) != sha_file(model_path / "hopper.json"):
        raise ValueError("Hopper (G) calibration map differs from the packaged map")
    package = hashlib.sha256()
    for path in sorted((source / "hopper_decisions").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            package.update(f"{path.relative_to(source)}\0{sha_file(path)}\n".encode())
    return {
        "checksums_sha256": sha_file(model_path / "CHECKSUMS.txt"),
        "verified_files": len(sums),
        "source_commit": head,
        "source_package_sha256": package.hexdigest(),
    }


def panel_answer(answer: dict[str, Any]) -> dict[str, Any]:
    if answer["type"] != "score":
        return answer
    probabilities = answer["probabilities"]
    return {
        "type": "score",
        "score": sum(int(level) * p for level, p in probabilities.items()),
        "probabilities": probabilities,
    }


def collect(
    *,
    model_path: Path,
    revision: str,
    source: Path,
    prompts: Path,
    output: Path,
    device: str = "cuda",
    max_items: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    rows = [
        json.loads(line)
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_items]
    model_path = model_path.resolve(strict=True)
    source = source.resolve(strict=True)
    identity = verify_release(model_path, revision, source)
    sys.path.insert(0, str(source))
    import torch
    from hopper_decisions.model import Decider

    decider = Decider(adapter=str(model_path), device=device, allow_slow_kernels=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            questions = row["questions"]
            torch.cuda.synchronize()
            started = time.perf_counter()
            answers = {}
            for name, question in questions.items():
                reply = decider.decide(
                    {"state": row["state"], "questions": {name: question}}
                )
                answers[name] = panel_answer(reply["answers"][name])
            torch.cuda.synchronize()
            target.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": answers,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "source_input_sha256": input_digest(row["state"], questions),
                        "backend": "hopper-decider",
                        "model_id": MODEL_ID,
                        "model_revision": revision,
                        "source_commit": identity["source_commit"],
                        "slow_kernels": sorted(map(str, decider.slow_kernels)),
                        "adapter_version": ADAPTER_VERSION,
                        "model_config_sha256": identity["checksums_sha256"],
                        "runtime_qualification": "unvalidated_rocm",
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            target.flush()
    return {"model_id": MODEL_ID, "revision": revision, "items": len(rows), **identity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                revision=args.model_revision,
                source=args.source_path,
                prompts=args.input,
                output=args.output,
                device=args.device,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
