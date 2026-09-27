"""Seal JPT-9B's complete, native, gold-free JevArena v3 peer predictions.

This binds the published llm2jev collector's receipts to the exact input
panel and to its earlier same-model development run. Run before scoring.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

from inference.jpt import (
    ADAPTER_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_REVISION,
    TEMPERATURE,
)
from inference.run import digest, file_digest, load_prompts

SCHEMA = "decision2-jpt9b-v3-peer-prediction-seal/1"
PANELS = {
    "typed-final": (
        1600,
        2000,
        "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    ),
    "css15": (
        6547,
        6547,
        "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    ),
}


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def seal(
    *,
    panel: str,
    prompts_path: Path,
    predictions_path: Path,
    manifest_path: Path,
    reference_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    if panel not in PANELS:
        raise ValueError("Unknown or unpinned peer panel")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    expected_items, expected_answers, expected_input_sha = PANELS[panel]
    if file_digest(prompts_path) != expected_input_sha:
        raise ValueError("Gold-free prompt bytes differ from the fixed v3 panel")
    prompts = load_prompts(prompts_path)
    if (
        len(prompts) != expected_items
        or sum(len(row["questions"]) for row in prompts) != expected_answers
    ):
        raise ValueError("Unexpected item or answer count")
    expected = {row["id"]: row for row in prompts}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    fixed_identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "temperature": TEMPERATURE,
        "adapter_version": ADAPTER_VERSION,
    }
    if any(manifest.get(key) != value for key, value in fixed_identity.items()):
        raise ValueError("Native manifest has a different model or adapter")
    if any(reference.get(key) != value for key, value in fixed_identity.items()):
        raise ValueError("Development reference used a different model or adapter")
    files = manifest.get("model_files_sha256")
    if not isinstance(files, dict) or files != reference.get("model_files_sha256"):
        raise ValueError("Model bytes differ from the pinned development peer")
    if (
        manifest.get("input_sha256") != expected_input_sha
        or manifest.get("output_sha256") != file_digest(predictions_path)
        or manifest.get("counts", {}).get("items") != expected_items
        or manifest.get("counts", {}).get("questions") != expected_answers
    ):
        raise ValueError("Native input, output or counts differ")
    seen: set[str] = set()
    invalid_answers = 0
    with predictions_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            item_id = row.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(
                    f"Unknown or duplicate prediction at line {line_number}"
                )
            seen.add(item_id)
            prompt = expected[item_id]
            if (
                row.get("source_input_sha256")
                != digest({"state": prompt["state"], "questions": prompt["questions"]})
                or row.get("backend") != "jpt-llm2jev-hf"
                or row.get("model_id") != MODEL_ID
                or row.get("model_revision") != MODEL_REVISION
                or row.get("adapter_version") != ADAPTER_VERSION
                or not isinstance(row.get("answers"), dict)
                or set(row["answers"]) != set(prompt["questions"])
            ):
                raise ValueError(f"Prediction identity or questions differ: {item_id}")
            for question_id, answer in row["answers"].items():
                if (
                    not isinstance(answer, dict)
                    or answer.get("type")
                    != prompt["questions"][question_id].get("type")
                    or not _finite(answer)
                ):
                    invalid_answers += 1
    if len(seen) != expected_items:
        raise ValueError("Native predictions are incomplete")
    result = {
        "schema": SCHEMA,
        "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "panel": panel,
        "post_key_same_panel": True,
        "gold_or_scores_read": False,
        **fixed_identity,
        "items": expected_items,
        "answer_slots": expected_answers,
        "obviously_invalid_answer_slots": invalid_answers,
        "input_sha256": expected_input_sha,
        "predictions_sha256": file_digest(predictions_path),
        "native_manifest_sha256": file_digest(manifest_path),
        "development_reference_manifest_sha256": file_digest(reference_manifest_path),
        "model_files_sha256": files,
        "sealer_sha256": file_digest(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=tuple(PANELS), required=True)
    for name in (
        "prompts_path",
        "predictions_path",
        "manifest_path",
        "reference_manifest_path",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = seal(**vars(args))
    print(json.dumps({"panel": result["panel"], "items": result["items"]}))


if __name__ == "__main__":
    main()
