"""Audit two fresh processes of one already selected native Decision candidate.

This is a gold-free runtime check. It never chooses a checkpoint or scores a
benchmark. The caller supplies a prospectively recorded numeric drift limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from publication.panel_parity import _point_and_values
from training.model.infer import load_prompts, prompt_input_sha256

VERSION = "decision2-candidate-repeat-smoke-v3/1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_predictions(
    path: Path, prompt_sha: str, model_id: str, revision: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prediction file must be regular")
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("predictions_sha256") != _sha(path)
        or manifest.get("input_sha256") != prompt_sha
        or manifest.get("model_id") != model_id
        or manifest.get("model_revision") != revision
        or not isinstance(manifest.get("model_sha256"), str)
        or not isinstance(manifest.get("adapter_sha256"), str)
        or not isinstance(manifest.get("calibration"), dict)
    ):
        raise ValueError("Prediction manifest does not bind the frozen inputs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest.get("input_items"):
        raise ValueError("Prediction count differs from manifest")
    return rows, manifest


def compare(
    prompts_path: Path,
    first_path: Path,
    second_path: Path,
    *,
    model_id: str,
    revision: str,
    max_drift: float,
) -> dict[str, Any]:
    if not math.isfinite(max_drift) or not 0 <= max_drift <= 0.02:
        raise ValueError("Prospective drift limit must be in [0, 0.02]")
    if len({prompts_path.resolve(), first_path.resolve(), second_path.resolve()}) != 3:
        raise ValueError("Prompt and prediction files must be distinct")
    prompts = load_prompts(prompts_path)
    if len(prompts) != 32:
        raise ValueError("This preflight requires the fixed 32 prompts")
    prompt_sha = _sha(prompts_path)
    first, first_manifest = _load_predictions(
        first_path, prompt_sha, model_id, revision
    )
    second, second_manifest = _load_predictions(
        second_path, prompt_sha, model_id, revision
    )
    if len(first) != len(prompts) or len(second) != len(prompts):
        raise ValueError("Incomplete prompt coverage")
    binding = (
        "model_sha256",
        "adapter_sha256",
        "calibration",
        "max_length",
        "adapter_version",
    )
    if any(first_manifest.get(key) != second_manifest.get(key) for key in binding):
        raise ValueError("Processes used different model, adapter, CAL or limits")
    drifts: list[float] = []
    changed: list[str] = []
    invalid = 0
    answers = 0
    for prompt, left, right in zip(prompts, first, second, strict=True):
        digest = prompt_input_sha256(prompt)
        questions = prompt["questions"]
        for row in (left, right):
            if (
                row.get("id") != prompt["id"]
                or row.get("source_input_sha256") != digest
                or row.get("model_sha256") != first_manifest["model_sha256"]
                or row.get("adapter_sha256") != first_manifest["adapter_sha256"]
                or set(row.get("answers", {})) != set(questions)
            ):
                raise ValueError("Prediction changed an input or model identity")
        if left.get("usage", {}).get("input_tokens") != right.get("usage", {}).get(
            "input_tokens"
        ):
            raise ValueError("Processes used different input token counts")
        for qid, question in questions.items():
            answers += 1
            left_point, left_values = _point_and_values(question, left["answers"][qid])
            right_point, right_values = _point_and_values(
                question, right["answers"][qid]
            )
            if left_point[0] != "ok" or right_point[0] != "ok":
                invalid += 1
            if left_point != right_point or len(left_values) != len(right_values):
                changed.append(f"{prompt['id']}:{qid}")
            if len(left_values) == len(right_values):
                drifts.extend(
                    abs(a - b) for a, b in zip(left_values, right_values, strict=True)
                )
    maximum = max(drifts, default=0.0)
    return {
        "schema_version": VERSION,
        "scope": "gold-free two-process repeatability, not a benchmark score",
        "model_id": model_id,
        "revision": revision,
        "prompts_sha256": prompt_sha,
        "prediction_sha256": {"first": _sha(first_path), "second": _sha(second_path)},
        "manifest_sha256": {
            "first": _sha(first_path.with_name(first_path.name + ".manifest.json")),
            "second": _sha(second_path.with_name(second_path.name + ".manifest.json")),
        },
        "items": len(prompts),
        "answers": answers,
        "invalid_answer_pairs": invalid,
        "category_mismatch_ids": changed,
        "max_probability_drift": maximum,
        "gate": {
            "invalid_answer_pairs": 0,
            "category_mismatch_n": 0,
            "max_probability_drift_lte": max_drift,
        },
        "gate_pass": invalid == 0 and not changed and maximum <= max_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-drift", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.prompts,
        args.first,
        args.second,
        model_id=args.model_id,
        revision=args.revision,
        max_drift=args.max_drift,
    )
    descriptor = os.open(
        args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps({"gate_pass": result["gate_pass"], "items": result["items"]}))


if __name__ == "__main__":
    main()
