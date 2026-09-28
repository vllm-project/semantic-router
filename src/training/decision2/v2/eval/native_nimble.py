"""Collect Bespoke-Nimble-9B-v2's native answers (bundled ``ParallelScorer``) on gold-free prompts.

The release's top-level ``inference`` module would collide with this repository's
``inference`` package, so this collector lives outside it. Every file listed in the
release's ``SHA256SUMS`` is verified, and the pinned Qwen3.5-9B base is loaded from the
local Hugging Face cache (offline) at the revision fixed in ``schema_config.json``.

Each item is scored as one native schema, as the release is used: Choice becomes an
``enum`` of the option keys, Noul a ``boolean`` and Score an integer-valued ``enum``
passed through ``score_fields``; option, level and true/false meanings go in
``choice_descriptions``. Structured states are rendered as compact JSON, as in the APUS
adapter. The release default temperature (2.179, transferred, not refit) is kept.
Requests over the native 8,192-token limit are rejected by the release without
truncation, and every question of that item is recorded as invalid; any other error
stops the run. The release targets CUDA; ROCm runs are labelled ``unvalidated_rocm``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from v2.eval.native_jet import attested_revision
from v2.eval.same_panel import input_digest, sha_file

MODEL_ID = "bespokelabs/Bespoke-Nimble-9B-v2"
MODEL_REVISION = "4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe"
BASE_ID = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER_VERSION = "nimble-v2-native-v1"


def verify_release(model_path: Path, revision: str) -> dict[str, Any]:
    if revision != MODEL_REVISION or attested_revision(model_path) != revision:
        raise ValueError("Nimble requires its attested, pinned Hugging Face revision")
    sums = {}
    for line in (model_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    for name in ("inference.py", "serving_schema.py", "adapter_model.safetensors"):
        if name not in sums:
            raise ValueError(f"Nimble SHA256SUMS lacks {name}")
    for name, expected in sums.items():
        if sha_file(model_path / name) != expected:
            raise ValueError(f"Nimble release file differs from SHA256SUMS: {name}")
    contract = json.loads(
        (model_path / "schema_config.json").read_text(encoding="utf-8")
    )
    if (contract.get("model"), contract.get("revision")) != (BASE_ID, BASE_REVISION):
        raise ValueError("Nimble contract names an unexpected base model")
    return {
        "sha256sums_sha256": sha_file(model_path / "SHA256SUMS"),
        "verified_files": len(sums),
        "max_length": contract["max_length"],
    }


def render_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def native_schema(questions: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    schema: dict[str, Any] = {}
    score_fields = []
    for name, question in questions.items():
        kind, criteria = question["type"], question.get("criteria")
        field: dict[str, Any] = {"description": question["instructions"]}
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"{name}: Choice needs at least two options")
            field.update(
                type="enum",
                choices=list(criteria),
                choice_descriptions={k: str(v) for k, v in criteria.items()},
            )
        elif kind == "noul":
            if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                raise ValueError(f"{name}: Noul mapping requires true/false criteria")
            field.update(
                type="boolean",
                choice_descriptions={
                    "true": str(criteria["true"]),
                    "false": str(criteria["false"]),
                },
            )
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"{name}: Score needs at least two levels")
            levels = [str(i) for i in range(len(criteria))]
            field.update(
                type="enum",
                choices=levels,
                choice_descriptions={
                    level: str(text) for level, text in zip(levels, criteria)
                },
            )
            score_fields.append(name)
        else:
            raise ValueError(f"{name}: unsupported question type {kind!r}")
        schema[name] = field
    return schema, tuple(score_fields)


def panel_answer(kind: str, field: dict[str, Any]) -> dict[str, Any]:
    probabilities = field["probabilities"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["true"]}
    if kind == "score":
        return {
            "type": "score",
            "score": field["expected_score"],
            "probabilities": probabilities,
        }
    return {
        "type": "choice",
        "choice": field["prediction"],
        "probabilities": probabilities,
    }


def is_length_rejection(exc: ValueError) -> bool:
    return "Nothing was truncated" in str(exc)


def collect(
    *,
    model_path: Path,
    revision: str,
    prompts: Path,
    output: Path,
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
    identity = verify_release(model_path, revision)
    sys.path.insert(0, str(model_path))
    import torch
    from inference import ParallelScorer

    engine = ParallelScorer(model_path)
    rejected = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            questions = row["questions"]
            schema, score_fields = native_schema(questions)
            torch.cuda.synchronize()
            started = time.perf_counter()
            error = None
            try:
                fields = engine.score(
                    render_state(row["state"]), schema, score_fields=score_fields
                )["fields"]
                answers = {
                    name: panel_answer(q["type"], fields[name])
                    for name, q in questions.items()
                }
            except ValueError as exc:
                if not is_length_rejection(exc):
                    raise
                answers = {
                    name: {"type": q["type"], "invalid_reason": "native_rejection"}
                    for name, q in questions.items()
                }
                error = str(exc)[:200]
                rejected += 1
            torch.cuda.synchronize()
            target.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "answers": answers,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "native_error": error,
                        "source_input_sha256": input_digest(row["state"], questions),
                        "backend": "nimble-parallel-scorer",
                        "model_id": MODEL_ID,
                        "model_revision": revision,
                        "base_revision": BASE_REVISION,
                        "temperature": engine.temperature,
                        "adapter_version": ADAPTER_VERSION,
                        "model_config_sha256": identity["sha256sums_sha256"],
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
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                model_path=args.model_path,
                revision=args.model_revision,
                prompts=args.input,
                output=args.output,
                max_items=args.max_items,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
