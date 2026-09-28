"""Collect Intern-Decision-0.8B's native answers from gold-free typed prompts.

Loads the release's bundled ``inference.py`` ``DecisionEngine`` (one causal forward
per request, candidate-symbol logits, released temperature) under a private module
name, because its file name collides with this package. Requests the engine rejects
natively (over 8,192 tokens, more than 62 options, more than 16 questions) become
invalid answers in the full denominator; nothing is truncated. The release pins
Transformers 5.14.1 on CUDA; ROCm runs are labelled ``unvalidated_rocm``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from .run import digest, file_digest, load_prompts, local_revision, synchronize

MODEL_ID = "internlm/Intern-Decision-0.8B"
MODEL_REVISION = "85a0cc5a99d67ea8d56dfe98115689212867171d"
ADAPTER_VERSION = "intern-decision-native-v1"


def model_fingerprint(model_path: Path) -> dict[str, str]:
    names = [
        "inference.py",
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    names += sorted(p.name for p in model_path.glob("*.safetensors"))
    missing = [n for n in names if not (model_path / n).is_file()]
    if missing or not any(n.endswith(".safetensors") for n in names):
        raise ValueError(f"Intern-Decision package incomplete: {missing}")
    return {name: file_digest(model_path / name) for name in names}


def load_engine(model_path: Path, device: str):
    spec = importlib.util.spec_from_file_location(
        "intern_decision_release_engine", model_path / "inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, module.DecisionEngine(str(model_path), device=device)


def invalid_answers(questions: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        name: {"type": question.get("type"), "invalid_reason": reason}
        for name, question in questions.items()
    }


def respond(engine: Any, row: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Return native answers, or invalid answers plus the native rejection text."""
    try:
        result = engine.predict({"state": row["state"], "questions": row["questions"]})
    except ValueError as exc:
        return invalid_answers(row["questions"], "native_rejection"), str(exc)[:200]
    answers = result.get("answers")
    if not isinstance(answers, dict) or answers.keys() != row["questions"].keys():
        raise ValueError(f"{row['id']}: native answers do not match question IDs")
    return answers, None


def collect(
    *, model_path: Path, revision: str, prompts: Path, output: Path, device: str
) -> dict[str, Any]:
    if revision != MODEL_REVISION:
        raise ValueError("Intern-Decision requires the pinned model revision")
    if output.exists():
        raise FileExistsError(output)
    rows = load_prompts(prompts)
    model_path = model_path.resolve(strict=True)
    if not local_revision(model_path, revision):
        raise ValueError("Intern-Decision download does not attest its pinned revision")
    files = model_fingerprint(model_path)
    fingerprint = digest(files)
    module, engine = load_engine(model_path, device)
    rejected = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for row in rows:
            synchronize(device)
            started = time.perf_counter()
            answers, error = respond(engine, row)
            synchronize(device)
            rejected += error is not None
            receipt = {
                "id": row["id"],
                "answers": answers,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "native_error": error,
                "source_input_sha256": digest(
                    {"state": row["state"], "questions": row["questions"]}
                ),
                "backend": "intern-decision-hf",
                "model_id": MODEL_ID,
                "model_revision": revision,
                "adapter_version": ADAPTER_VERSION,
                "model_config_sha256": fingerprint,
                "runtime_qualification": "unvalidated_rocm",
            }
            target.write(
                json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n"
            )
            target.flush()
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "temperature": module.DEFAULT_TEMPERATURE,
        "items": len(rows),
        "native_rejections": rejected,
        "model_files_sha256": files,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = collect(
        model_path=args.model_path,
        revision=args.model_revision,
        prompts=args.input,
        output=args.output,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
