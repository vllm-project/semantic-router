"""Seal gold-free 0.6B same-panel predictions before post-key scoring.

This receipt proves the named prediction bytes were fixed before this run's
scoring. It deliberately does not claim that the panel labels were never
accessed earlier in the project.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest, load_prompts

PANEL_FIELDS = {
    "typed": ("typed_final_prompts_sha256", 1600),
    "css": ("css_evaluation_prompts_sha256", 6547),
    "public": ("jevbench_public_prompts_sha256", 231),
}
MODEL_STEMS = {"Kai1": "kai1", "Kai2": "kai2", "Bosun-v3.1-0.6B": "bosun"}
SCHEMA = "decision2-jevarena-v3-postkey-prediction-seal/1"


def _predictions(
    path: Path, prompts: list[dict[str, Any]], model: dict[str, Any]
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked prediction file: {path.name}")
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in prompts
    }
    if len(expected) != len(prompts):
        raise ValueError("Prompt IDs are not unique")
    seen = set()
    answer_slots = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            item_id = row.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(
                    f"Missing, unknown or repeated prediction at line {number}"
                )
            source_hash, question_ids = expected[item_id]
            if (
                row.get("model_id") != model["model_id"]
                or row.get("model_revision") != model["weight_revision"]
                or row.get("adapter_version") != model["native_adapter"]
                or row.get("source_input_sha256") != source_hash
                or not isinstance(row.get("answers"), dict)
                or set(row["answers"]) != question_ids
            ):
                raise ValueError(
                    f"Prediction provenance or answer slots differ at {number}"
                )
            seen.add(item_id)
            answer_slots += len(question_ids)
    if seen != set(expected):
        raise ValueError(f"Prediction file is incomplete: {path.name}")
    return {
        "file": path.name,
        "sha256": file_digest(path),
        "items": len(seen),
        "answer_slots": answer_slots,
        "model_id": model["model_id"],
        "weight_revision": model["weight_revision"],
        "adapter_version": model["native_adapter"],
    }


def seal(
    roster_path: Path,
    typed_prompts: Path,
    css_prompts: Path,
    public_prompts: Path,
    prediction_dir: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if (
        roster.get("schema") != "decision2-jevarena-v3-postkey-roster/1"
        or roster.get("status") != "prospective_same_panel_after_project_label_access"
        or roster.get("not_a_never_unsealed_blind_test") is not True
        or set(model["name"] for model in roster["model_roster"]) != set(MODEL_STEMS)
    ):
        raise ValueError("Unexpected post-key roster or chronology")
    paths = {"typed": typed_prompts, "css": css_prompts, "public": public_prompts}
    panels = {}
    for panel, path in paths.items():
        field, cardinality = PANEL_FIELDS[panel]
        if file_digest(path) != roster["panel"][field]:
            raise ValueError(f"{panel}: prompt bytes differ from the roster")
        prompts = load_prompts(path)
        if len(prompts) != cardinality:
            raise ValueError(f"{panel}: prompt cardinality differs")
        panels[panel] = {"path": path, "prompts": prompts, "sha256": file_digest(path)}
    records = {}
    for model in roster["model_roster"]:
        stem = MODEL_STEMS[model["name"]]
        records[stem] = {
            panel: _predictions(
                prediction_dir / f"{stem}.{panel}.jsonl", data["prompts"], model
            )
            for panel, data in panels.items()
        }
    receipt = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "roster_sha256": file_digest(roster_path),
        "code_sha256": file_digest(Path(__file__)),
        "post_key": True,
        "claim": "predictions fixed before this run's scoring, not virgin blind",
        "prompt_sha256": {panel: data["sha256"] for panel, data in panels.items()},
        "predictions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    output.chmod(0o600)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "roster",
        "typed_prompts",
        "css_prompts",
        "public_prompts",
        "prediction_dir",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    receipt = seal(
        args.roster,
        args.typed_prompts,
        args.css_prompts,
        args.public_prompts,
        args.prediction_dir,
        args.output,
    )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "roster_sha256": receipt["roster_sha256"],
                "seal_sha256": file_digest(args.output),
                "items_per_model": sum(
                    x["items"] for x in receipt["predictions"]["kai1"].values()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
