"""Jointly seal fresh Lux 1.0 r4 native predictions before any scoring."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

from inference.run import (
    OVER_BUDGET_ADAPTER_VERSION,
    completed_rows,
    file_digest,
    load_prompts,
    local_revision,
)

SCHEMA = "decision2-lux9b-v3-public231-r4-joint-seal/1"
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
    "public231": (
        231,
        231,
        "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
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


def validate_panel(
    panel: str, prompts_path: Path, predictions_path: Path
) -> dict[str, Any]:
    expected_items, expected_answers, expected_input_sha = PANELS[panel]
    if file_digest(prompts_path) != expected_input_sha:
        raise ValueError(f"{panel}: gold-free prompt bytes differ from the fixed panel")
    prompts = load_prompts(prompts_path)
    if (
        len(prompts) != expected_items
        or sum(len(row["questions"]) for row in prompts) != expected_answers
    ):
        raise ValueError(f"{panel}: unexpected original or answer-slot count")
    completed = completed_rows(
        predictions_path,
        prompts,
        "lux",
        REVISION,
        CONFIG_SHA256,
        model_id=MODEL_ID,
        revision_attested=True,
        adapter_version=OVER_BUDGET_ADAPTER_VERSION,
    )
    if len(completed) != expected_items:
        raise ValueError(f"{panel}: native predictions are incomplete")
    by_id = {row["id"]: row for row in prompts}
    invalid_answers = 0
    over_budget_rows = 0
    over_budget_answer_slots = 0
    with predictions_path.open(encoding="utf-8") as source:
        for line in source:
            prediction = json.loads(line)
            if prediction.get("runtime_matches_validated") is not True:
                raise ValueError(f"{panel}: native runtime did not match release")
            if prediction.get("model") != "Decision-1.0-Lux":
                raise ValueError(f"{panel}: native model identity mismatch")
            item = by_id[prediction["id"]]
            error = prediction.get("native_error")
            if error is not None:
                if (
                    not isinstance(error, dict)
                    or set(error)
                    != {"kind", "question_id", "input_tokens", "max_length"}
                    or error["kind"] != "native_input_over_budget"
                    or error["question_id"] not in item["questions"]
                    or type(error["input_tokens"]) is not int
                    or type(error["max_length"]) is not int
                    or error["max_length"] < 1
                    or error["input_tokens"] <= error["max_length"]
                    or any(
                        value is not None for value in prediction["answers"].values()
                    )
                ):
                    raise ValueError(f"{panel}: malformed native over-budget receipt")
                over_budget_rows += 1
                over_budget_answer_slots += len(item["questions"])
            for key, answer in prediction["answers"].items():
                if (
                    not isinstance(answer, dict)
                    or answer.get("type") != item["questions"][key].get("type")
                    or not _finite(answer)
                ):
                    invalid_answers += 1
    return {
        "originals": expected_items,
        "answer_slots": expected_answers,
        "input_sha256": expected_input_sha,
        "predictions_sha256": file_digest(predictions_path),
        "native_over_budget_originals": over_budget_rows,
        "native_over_budget_answer_slots": over_budget_answer_slots,
        "obviously_invalid_answer_slots": invalid_answers,
    }


def seal(
    *,
    model_path: Path,
    run_lock: Path,
    typed_prompts: Path,
    typed_predictions: Path,
    css_prompts: Path,
    css_predictions: Path,
    public_prompts: Path,
    public_predictions: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    model_path = model_path.resolve(strict=True)
    if file_digest(model_path / "bundle-manifest.json") != CONFIG_SHA256:
        raise ValueError("Lux package manifest differs")
    if not local_revision(model_path, REVISION):
        raise ValueError("Lux package revision is not attested")
    panels = {
        "typed-final": validate_panel("typed-final", typed_prompts, typed_predictions),
        "css15": validate_panel("css15", css_prompts, css_predictions),
        "public231": validate_panel("public231", public_prompts, public_predictions),
    }
    result = {
        "schema": SCHEMA,
        "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "post_key_same_panel": True,
        "gold_or_scores_read": False,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "bundle_manifest_sha256": CONFIG_SHA256,
        "adapter_version": OVER_BUDGET_ADAPTER_VERSION,
        "run_lock_sha256": file_digest(run_lock),
        "collector_sha256": file_digest(
            Path(__file__).parents[1] / "inference" / "run.py"
        ),
        "sealer_sha256": file_digest(Path(__file__)),
        "panels": panels,
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
    for name in (
        "model_path",
        "run_lock",
        "typed_prompts",
        "typed_predictions",
        "css_prompts",
        "css_predictions",
        "public_prompts",
        "public_predictions",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = seal(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "panels": {
                    key: value["originals"] for key, value in result["panels"].items()
                },
                "native_over_budget_originals": {
                    key: value["native_over_budget_originals"]
                    for key, value in result["panels"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
