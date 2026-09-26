"""Collect pinned JevK5 native typed readout on gold-free benchmark prompts.

The released JevK5 runtime renders the SemIf prompt, reads letter logits, and
applies its own model temperatures. This adapter preserves that behavior and
adds source/model/runtime receipts for the unified Decision benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

ADAPTER_VERSION = "jevk5-native-eager-v2-admission"
RUNTIME_REVISION = "1e5ae1b533b9eb80c0cbe3fbd010607d0b4e26ae"
SPECS = {
    "2b": ("alibiserikbay/JevK5-2B", "7922d1f55df137b72ef763fced56fd09efc5e99d"),
    "4b": ("alibiserikbay/JevK5", "c4f7fdb3aeab5582336406e78d3bef11bf98833d"),
    "9b": ("alibiserikbay/JevK5-9B", "d6521a18a86999190e9d775c915af3d6d6772fc4"),
}
REQUIRED_MODEL_FILES = {
    "README.md",
    "chat_template.jinja",
    "config.json",
    "jevk5_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}


def verify_release(path: Path, size: str, revision: str) -> dict[str, Any]:
    if (
        size not in SPECS
        or revision != SPECS[size][1]
        or not local_revision(path, revision)
    ):
        raise ValueError(
            "JevK5 requires its pinned, locally attested Hugging Face revision"
        )
    listing = path / "SHA256SUMS"
    if listing.exists():
        hashes: dict[str, str] = {}
        for line in listing.read_text(encoding="utf-8").splitlines():
            expected, separator, name = line.partition("  ")
            relative = Path(name)
            if (
                separator != "  "
                or len(expected) != 64
                or name in hashes
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError("Malformed or unsafe JevK5 SHA256SUMS")
            file = (path / relative).resolve()
            if (
                not file.is_relative_to(path.resolve())
                or not file.is_file()
                or file_digest(file) != expected
            ):
                raise ValueError(f"JevK5 release file differs from SHA256SUMS: {name}")
            hashes[name] = expected
        if not set(hashes) >= REQUIRED_MODEL_FILES:
            raise ValueError("JevK5 SHA256SUMS omits required release files")
        listing_sha = file_digest(listing)
    else:
        if size != "2b":
            raise ValueError("JevK5 4B and 9B require their released SHA256SUMS")
        hashes = {}
        for name in sorted(REQUIRED_MODEL_FILES):
            file = path / name
            if not file.is_file():
                raise FileNotFoundError(file)
            hashes[name] = file_digest(file)
        listing_sha = None
    return {
        "release_files_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "model_weight_sha256": hashes["model.safetensors"],
        "model_config_sha256": hashes["jevk5_config.json"],
        "release_manifest_sha256": listing_sha,
    }


def verify_runtime(path: Path) -> dict[str, str]:
    if (
        not (path / "jevk5/runtime.py").is_file()
        or not (path / "jevk5/prompt.py").is_file()
    ):
        raise FileNotFoundError("Pinned JevK5 runtime source is missing")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != RUNTIME_REVISION or dirty:
        raise ValueError("JevK5 runtime differs from the pinned clean source revision")
    return {
        "runtime_revision": head,
        "runtime_source_sha256": digest(
            {
                name: file_digest(path / name)
                for name in ("jevk5/runtime.py", "jevk5/prompt.py")
            }
        ),
    }


def load_native(path: Path, runtime_path: Path, device: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(runtime_path))
    import torch
    from jevk5 import JevK5

    return JevK5(str(path), device=device, dtype=torch.bfloat16, graphs=False)


def shared_answer(question: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    """Verify JevK5's Score is the native probability-weighted expectation."""
    kind = question["type"]
    if not isinstance(native, dict) or native.get("type") != kind:
        raise ValueError("JevK5 returned the wrong native answer type")
    tokens = native.get("input_tokens")
    if type(tokens) is not int or tokens < 1:
        raise ValueError("JevK5 omitted the native input token count")
    confidence = native.get("confidence")
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("JevK5 returned invalid confidence")
    if kind == "noul":
        p_true = native.get("noul")
        if (
            type(p_true) not in (int, float)
            or not math.isfinite(p_true)
            or not 0 <= p_true <= 1
        ):
            raise ValueError("JevK5 returned invalid Noul probability")
        expected_confidence = max(p_true, 1 - p_true)
    elif kind in {"choice", "score"}:
        probabilities = native.get("probabilities")
        expected_keys = (
            set(question["criteria"])
            if kind == "choice"
            else {str(i) for i in range(len(question["criteria"]))}
        )
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != expected_keys
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities.values()
            )
            or abs(sum(probabilities.values()) - 1) > 0.02
        ):
            raise ValueError("JevK5 returned an invalid option probability map")
        expected_confidence = max(probabilities.values())
        if kind == "choice":
            selected = native.get("choice")
            if (
                selected not in probabilities
                or probabilities[selected] < expected_confidence - 0.02
            ):
                raise ValueError("JevK5 choice conflicts with its probabilities")
        else:
            expected_score = sum(
                int(key) * value for key, value in probabilities.items()
            )
            score = native.get("score")
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or abs(score - expected_score) > 1e-6
            ):
                raise ValueError("JevK5 Score is not its native expected value")
    else:
        raise ValueError("Unsupported JevK5 question type")
    if abs(confidence - expected_confidence) > 1e-6:
        raise ValueError("JevK5 confidence conflicts with its native probabilities")
    return native


def native_admission_reason(error: ValueError) -> str | None:
    """Map documented request-size refusals to explicit invalid benchmark rows."""
    message = str(error).lower()
    if ("token" in message or "context" in message) and any(
        marker in message
        for marker in ("16384", "exceed", "too long", "maximum", "limit")
    ):
        return "context_overflow"
    if ("option" in message or "candidate" in message) and any(
        marker in message for marker in ("exceed", "too many", "maximum", "limit")
    ):
        return "candidate_limit"
    return None


def completed_rows(
    path: Path, rows: list[dict[str, Any]], identity: dict[str, Any]
) -> set[str]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in rows
    }
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            item_id = item.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(f"{path}:{line_number}: unknown or duplicate ID")
            if any(item.get(key) != value for key, value in identity.items()):
                raise ValueError(f"{path}:{line_number}: stale JevK5 identity")
            if (
                item.get("source_input_sha256") != expected[item_id][0]
                or not isinstance(item.get("answers"), dict)
                or set(item["answers"]) != expected[item_id][1]
            ):
                raise ValueError(
                    f"{path}:{line_number}: stale or incomplete JevK5 answer"
                )
            seen.add(item_id)
    return seen


def collect(
    *,
    size: str,
    model_path: Path,
    runtime_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if size not in SPECS or (max_items is not None and max_items < 1):
        raise ValueError("Unknown JevK5 size or invalid max_items")
    rows = load_prompts(prompts)
    model_path = model_path.resolve(strict=True)
    runtime_path = runtime_path.resolve(strict=True)
    release = verify_release(model_path, size, revision)
    runtime = verify_runtime(runtime_path)
    identity = {
        "backend": "jevk5-native-eager",
        "model_id": SPECS[size][0],
        "model_revision": revision,
        "revision_attested": True,
        "adapter_version": ADAPTER_VERSION,
        **release,
        **runtime,
    }
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        seen = completed_rows(output, rows, identity)
    else:
        seen = set()
    remaining = [row for row in rows if row["id"] not in seen]
    if max_items is not None:
        remaining = remaining[:max_items]
    if remaining:
        native = load_native(model_path, runtime_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            answers: dict[str, dict[str, Any]] = {}
            invalid_reasons: dict[str, str] = {}
            for name, question in row["questions"].items():
                try:
                    answers[name] = shared_answer(
                        question, native.decide(row["state"], question)
                    )
                except ValueError as error:
                    reason = native_admission_reason(error)
                    if reason is None:
                        raise
                    answers[name] = {"type": question["type"], "error": reason}
                    invalid_reasons[name] = reason
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if set(answers) != set(row["questions"]) or not math.isfinite(latency_ms):
                raise ValueError(f"{row['id']}: malformed JevK5 native response")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "usage": {
                    "input_tokens": sum(
                        answer.get("input_tokens", 0) for answer in answers.values()
                    ),
                    "output_tokens": 0,
                },
                "latency_ms": latency_ms,
                "source_input_sha256": digest(payload),
                "model": f"{SPECS[size][0]}@{revision}",
                "runtime_qualification": "pytorch_bf16_rocm_unvalidated",
                "invalid_reason": invalid_reasons or None,
                **identity,
            }
            target.write(
                json.dumps(
                    receipt, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            target.flush()
    return {
        "input_items": len(rows),
        "previously_completed": len(seen),
        "collected_now": len(remaining),
        "output": str(output),
        **identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(SPECS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                size=args.size,
                model_path=args.model_path,
                runtime_path=args.runtime_path,
                revision=args.model_revision,
                prompts=args.input,
                output=args.output,
                device=args.device,
                resume=args.resume,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
