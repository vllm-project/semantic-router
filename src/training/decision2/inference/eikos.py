"""Collect the published Eikos-4B native typed readout on gold-free prompts.

The checkpoint's own ``serve.Decider`` and ``letter_adapter.LetterAdapter``
provide prompt rendering, single-pass letter logits, and released calibration.
This collector only binds those outputs to the shared benchmark receipt format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

MODEL_ID = "caiovicentino1/Eikos-4B"
REVISION = "582ffb13f19a4da3f455e3db198584190bd7755b"
ADAPTER_VERSION = "eikos-native-letter-v1"


def verify_release(path: Path, revision: str) -> dict[str, str]:
    if revision != REVISION or not local_revision(path, revision):
        raise ValueError("Eikos requires its attested, pinned Hugging Face revision")
    listing = path / "SHA256SUMS"
    if not listing.is_file():
        raise FileNotFoundError(listing)
    names: set[str] = set()
    for line in listing.read_text(encoding="utf-8").splitlines():
        expected, separator, name = line.partition("  ")
        if separator != "  " or len(expected) != 64 or name in names:
            raise ValueError("Malformed Eikos release SHA256SUMS")
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path.resolve()) or not candidate.is_file():
            raise ValueError(f"Unsafe or missing Eikos release file: {name}")
        if file_digest(candidate) != expected:
            raise ValueError(f"Eikos release file differs from SHA256SUMS: {name}")
        names.add(name)
    required = {
        "decision_config.json",
        "calib.json",
        "decision_core.py",
        "letter_adapter.py",
        "serve.py",
        "config.json",
        "model.safetensors.index.json",
    }
    if not required.issubset(names):
        raise ValueError("Eikos release roster lacks native code or configuration")
    config = json.loads((path / "decision_config.json").read_text(encoding="utf-8"))
    if config != {
        "prompt_version": "letter-v1-semif",
        "readout": "letter-logit",
        "calib": "calib.json",
        "max_one_pass": 100,
    }:
        raise ValueError("Unexpected Eikos native decision contract")
    return {
        "release_manifest_sha256": file_digest(listing),
        "model_config_sha256": file_digest(path / "decision_config.json"),
    }


def load_native(path: Path, device: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PROMPT_STYLE"] = "semif"
    sys.path.insert(0, str(path))
    import decision_core
    from serve import Decider

    decision_core.set_max_one_pass(100)
    options = argparse.Namespace(
        model=str(path),
        adapter=None,
        temp=1.0,
        calib=str(path / "calib.json"),
        max_tokens=16000,
        verify_budget=0,
        vllm_url=None,
        sglang_url=None,
        device=device,
        sym=False,
    )
    return Decider(options)


def shared_answer(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Preserve native probabilities; express Score as its expected value.

    Eikos returns the modal level in ``score`` and the probability-weighted
    mean in ``expected``. The shared Score schema uses the mean in ``score``
    and obtains the class decision from the published distribution's mode.
    """
    if question["type"] != "score":
        return answer
    expected = answer.get("expected")
    if not isinstance(expected, (float, int)) or not math.isfinite(expected):
        raise ValueError("Eikos returned a nonfinite Score expectation")
    return {
        **answer,
        "native_score": answer["score"],
        "score": float(expected),
        "score_projection": "native_expected_from_level_probabilities",
    }


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
                raise ValueError(f"{path}:{line_number}: stale Eikos identity")
            if (
                item.get("source_input_sha256") != expected[item_id][0]
                or not isinstance(item.get("answers"), dict)
                or set(item["answers"]) != expected[item_id][1]
            ):
                raise ValueError(
                    f"{path}:{line_number}: stale or incomplete Eikos answer"
                )
            seen.add(item_id)
    return seen


def collect(
    *,
    model_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
    device: str = "cuda:0",
    resume: bool = False,
    max_items: int | None = None,
) -> dict[str, Any]:
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be positive")
    rows = load_prompts(prompts)
    model_path = model_path.resolve(strict=True)
    release = verify_release(model_path, revision)
    identity = {
        "backend": "eikos",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "revision_attested": True,
        "adapter_version": ADAPTER_VERSION,
        **release,
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
    native = load_native(model_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if output.exists() else "x", encoding="utf-8") as target:
        for row in remaining:
            payload = {"state": row["state"], "questions": row["questions"]}
            synchronize(device)
            started = time.perf_counter()
            try:
                result = native.decide_all(**payload)
                answers = {
                    name: shared_answer(row["questions"][name], native_answer)
                    for name, (native_answer, _) in result.items()
                }
                usage = {
                    "input_tokens": sum(count for _, count in result.values()),
                    "output_tokens": 0,
                }
                invalid_reason = None
            except ValueError as error:
                if "tokens > 16000" not in str(error):
                    raise
                invalid_reason = "context_overflow"
                answers = {
                    name: {"type": question["type"], "error": invalid_reason}
                    for name, question in row["questions"].items()
                }
                usage = None
            synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000
            if set(answers) != set(row["questions"]) or not math.isfinite(latency_ms):
                raise ValueError(f"{row['id']}: malformed Eikos native response")
            receipt = {
                "id": row["id"],
                "answers": answers,
                "usage": usage,
                "latency_ms": latency_ms,
                "source_input_sha256": digest(payload),
                "model": f"{MODEL_ID}@{revision}",
                "runtime_qualification": "pytorch_bf16_rocm_unvalidated",
                "invalid_reason": invalid_reason,
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
    parser.add_argument("--model-path", type=Path, required=True)
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
                model_path=args.model_path,
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
