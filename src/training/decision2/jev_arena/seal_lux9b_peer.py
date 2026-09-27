"""Seal complete native Lux 1.0 predictions on the fixed JevArena v3 panels."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

from inference.run import completed_rows, file_digest, load_prompts, local_revision

SCHEMA = "decision2-lux9b-v3-control-prediction-seal/1"
MODEL_ID = "llm-semantic-router/Decision-1.0-Lux-9B"
REVISION = "bd45a30aee8c84032791c245c70f86dee5389cc8"
CONFIG_SHA256 = "985ade73c509399291d60b5f98e8bbbbe99c0ee0efe611a5f84604f71420e0fd"
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
    model_path: Path,
    prompts_path: Path,
    predictions_path: Path,
    output: Path,
) -> dict[str, Any]:
    if panel not in PANELS:
        raise ValueError("Unknown or unpinned control panel")
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
    model_path = model_path.resolve(strict=True)
    if file_digest(
        model_path / "bundle-manifest.json"
    ) != CONFIG_SHA256 or not local_revision(model_path, REVISION):
        raise ValueError("Lux model manifest or pinned revision differs")
    completed = completed_rows(
        predictions_path,
        prompts,
        "lux",
        REVISION,
        CONFIG_SHA256,
        model_id=MODEL_ID,
        revision_attested=True,
    )
    if len(completed) != expected_items:
        raise ValueError("Native predictions are incomplete")
    by_id = {row["id"]: row for row in prompts}
    invalid_answers = 0
    with predictions_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("runtime_matches_validated") is not True:
                raise ValueError("Native runtime did not match the released profile")
            for key, answer in row["answers"].items():
                if (
                    not isinstance(answer, dict)
                    or answer.get("type")
                    != by_id[row["id"]]["questions"][key].get("type")
                    or not _finite(answer)
                ):
                    invalid_answers += 1
    result = {
        "schema": SCHEMA,
        "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "panel": panel,
        "post_key_same_panel": True,
        "gold_or_scores_read": False,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "bundle_manifest_sha256": CONFIG_SHA256,
        "items": expected_items,
        "answer_slots": expected_answers,
        "obviously_invalid_answer_slots": invalid_answers,
        "input_sha256": expected_input_sha,
        "predictions_sha256": file_digest(predictions_path),
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
    for name in ("model_path", "prompts_path", "predictions_path", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = seal(**vars(args))
    print(json.dumps({"panel": result["panel"], "items": result["items"]}))


if __name__ == "__main__":
    main()
