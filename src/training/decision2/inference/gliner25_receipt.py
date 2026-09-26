"""Bind a complete GLiNER2.5 baseline prediction file to model and prompt bytes."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from . import gliner25
from .run import digest, file_digest, load_prompts

SCHEMA_VERSION = "decision2-gliner25-prediction-manifest/1"


def audit(
    rows: list[dict[str, Any]], predictions: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in rows
    }
    seen = set()
    reasons = Counter()
    with predictions.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            item_id = item.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(
                    f"{predictions}:{line_number}: unknown or duplicate ID"
                )
            if any(item.get(key) != value for key, value in identity.items()):
                raise ValueError(f"{predictions}:{line_number}: wrong model or adapter")
            if (
                item.get("source_input_sha256") != expected[item_id][0]
                or not isinstance(item.get("answers"), dict)
                or set(item["answers"]) != expected[item_id][1]
            ):
                raise ValueError(
                    f"{predictions}:{line_number}: stale or incomplete answer"
                )
            reason = item.get("invalid_reason")
            if reason is not None and reason != "context_overflow":
                raise ValueError(
                    f"{predictions}:{line_number}: unexplained invalid answer"
                )
            if reason is None and any(
                "error" in answer for answer in item["answers"].values()
            ):
                raise ValueError(
                    f"{predictions}:{line_number}: uncounted invalid answer"
                )
            if reason == "context_overflow" and not any(
                answer.get("error") == reason
                and answer.get("native_input_tokens", 0)
                > answer.get("native_max_positions", 0)
                for answer in item["answers"].values()
            ):
                raise ValueError(
                    f"{predictions}:{line_number}: overflow lacks native evidence"
                )
            reasons[reason or "valid"] += 1
            seen.add(item_id)
    if seen != set(expected):
        raise ValueError(f"Prediction file is incomplete: {len(seen)}/{len(expected)}")
    return {
        "items": len(rows),
        "valid": reasons["valid"],
        "invalid": len(rows) - reasons["valid"],
        "invalid_reasons": {
            key: value for key, value in sorted(reasons.items()) if key != "valid"
        },
    }


def write(
    *, model_path: Path, prompts: Path, predictions: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    release = gliner25.verify_release(
        model_path.resolve(strict=True), gliner25.REVISION
    )
    rows = load_prompts(prompts)
    identity = {
        "backend": "gliner25",
        "model_id": gliner25.MODEL_ID,
        "model_revision": gliner25.REVISION,
        "revision_attested": True,
        "library_commit": gliner25.LIBRARY_COMMIT,
        "adapter_version": gliner25.ADAPTER_VERSION,
        "prompt_projection": "state-text; instruction-and-criteria-native-schema",
        **release,
    }
    counts = audit(rows, predictions, identity)
    import gliner2
    import torch
    import transformers

    if (
        gliner2.__version__ != "2.0.0"
        or os.environ.get("GLINER2_SOURCE_COMMIT") != gliner25.LIBRARY_COMMIT
    ):
        raise RuntimeError("Runtime does not match the pinned GLiNER source")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": file_digest(prompts),
        "predictions_sha256": file_digest(predictions),
        "collector_sha256": file_digest(Path(gliner25.__file__)),
        "model_identity": identity,
        "counts": counts,
        "runtime": {
            "gliner2": gliner2.__version__,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(
            receipt,
            target,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        target.write("\n")
    return {"output": str(output), "manifest_sha256": file_digest(output), **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write(
                model_path=args.model_path,
                prompts=args.input,
                predictions=args.predictions,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
