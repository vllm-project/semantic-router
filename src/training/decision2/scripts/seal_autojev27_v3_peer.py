"""Seal AutoJev's three complete gold-free peer panels before scoring labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference.autojev27 import ADAPTER_VERSION, BACKEND, MODEL_ID, MODEL_REVISION
from inference.run import digest, file_digest, load_prompts

SCHEMA = "decision2-autojev27-v3-peer-prediction-seal/1"
MODEL_SHA = "d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2"
SOURCE_SHA = "550ccd857350c6771a1de03e4bcba9fb4412247b58bcdc9e8ac03de2ab9641a5"
CONFIG_SHA = "bacbcbb281a53af5ef5cc6c9028601097d155bf981129f18a727219517921dcd"
PANELS = {
    "typed": (
        1600,
        2000,
        "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    ),
    "css": (
        6547,
        6547,
        "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    ),
    "public": (
        231,
        231,
        "642d3fac1b6521fe33df72f9228e4e364b7be7ea277893207f97da5bc75ddd",
    ),
}


def audit_panel(
    *,
    prompts: Path,
    predictions: Path,
    expected_items: int,
    expected_questions: int,
    expected_prompt_sha256: str,
) -> dict[str, Any]:
    if file_digest(prompts) != expected_prompt_sha256:
        raise ValueError("Prompt bytes differ from preregistration")
    inputs = load_prompts(prompts)
    if (
        len(inputs) != expected_items
        or sum(len(row["questions"]) for row in inputs) != expected_questions
    ):
        raise ValueError("Prompt panel has wrong item or question denominator")
    outputs = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
    ]
    if len(outputs) != expected_items:
        raise ValueError("Incomplete or extra native prediction rows")
    invalid = 0
    for prompt, row in zip(inputs, outputs, strict=True):
        if (
            row.get("id") != prompt["id"]
            or row.get("source_input_sha256")
            != digest({"state": prompt["state"], "questions": prompt["questions"]})
            or row.get("model_id") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
            or row.get("revision_attested") is not True
            or row.get("backend") != BACKEND
            or row.get("adapter_version") != ADAPTER_VERSION
            or row.get("native_model_sha256") != MODEL_SHA
            or row.get("runtime_source_sha256") != SOURCE_SHA
            or row.get("model_config_sha256") != CONFIG_SHA
            or not isinstance(row.get("answers"), dict)
            or set(row["answers"]) != set(prompt["questions"])
        ):
            raise ValueError("Native prediction differs from frozen model or input")
        for key, question in prompt["questions"].items():
            answer = row["answers"][key]
            if not isinstance(answer, dict) or answer.get("type") != question["type"]:
                raise ValueError("Native answer type is missing or changed")
            if "error" in answer:
                if set(answer) != {"type", "error"} or answer["error"] not in {
                    "context_overflow",
                    "candidate_limit",
                }:
                    raise ValueError("Unexpected native invalid-answer encoding")
                invalid += 1
    return {
        "prompt_sha256": expected_prompt_sha256,
        "prediction_sha256": file_digest(predictions),
        "items": expected_items,
        "questions": expected_questions,
        "declared_native_invalid_answers": invalid,
    }


def seal(panels: dict[str, tuple[Path, Path]], image_id: str) -> dict[str, Any]:
    if set(panels) != set(PANELS) or not image_id.startswith("sha256:"):
        raise ValueError("All three panels and an immutable runtime image are required")
    result = {}
    for name, (prompts, predictions) in panels.items():
        count, questions, prompt_sha = PANELS[name]
        result[name] = audit_panel(
            prompts=prompts,
            predictions=predictions,
            expected_items=count,
            expected_questions=questions,
            expected_prompt_sha256=prompt_sha,
        )
    return {
        "schema_version": SCHEMA,
        "status": "all_gold_free_predictions_sealed_before_scoring",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "native_model_sha256": MODEL_SHA,
        "runtime_source_sha256": SOURCE_SHA,
        "adapter_version": ADAPTER_VERSION,
        "image_id": image_id,
        "panels": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PANELS:
        parser.add_argument(f"--{name}-prompts", required=True, type=Path)
        parser.add_argument(f"--{name}-predictions", required=True, type=Path)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    panels = {
        name: (getattr(args, f"{name}_prompts"), getattr(args, f"{name}_predictions"))
        for name in PANELS
    }
    receipt = seal(panels, args.image_id)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps({"status": receipt["status"], "panels": list(receipt["panels"])}))


if __name__ == "__main__":
    main()
