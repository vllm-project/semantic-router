"""Summarize the prospectively sealed, post-key 0.6B same-panel comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from inference.run import file_digest
from jev_arena.arena_v3 import _css, _typed
from jev_arena.seal_postkey_06b import MODEL_STEMS

SCHEMA = "decision2-jevarena-v3-postkey-06b-rank/1"


def _report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or linked score report: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def rank(roster_path: Path, seal_path: Path, reports_dir: Path) -> dict[str, Any]:
    roster = _report(roster_path)
    seal = _report(seal_path)
    if (
        seal.get("schema") != "decision2-jevarena-v3-postkey-prediction-seal/1"
        or seal.get("roster_sha256") != file_digest(roster_path)
        or seal.get("post_key") is not True
        or roster.get("not_a_never_unsealed_blind_test") is not True
    ):
        raise ValueError("Missing or mismatched post-key prediction seal")
    rows = []
    report_hashes = {}
    for model in roster["model_roster"]:
        stem = MODEL_STEMS[model["name"]]
        paths = {
            panel: reports_dir / f"{stem}.{panel}.score.json"
            for panel in ("typed", "css", "public")
        }
        reports = {panel: _report(path) for panel, path in paths.items()}
        report_hashes[stem] = {
            panel: file_digest(path) for panel, path in paths.items()
        }
        frozen = seal["predictions"][stem]
        if any(
            reports[panel].get("predictions_sha256") != frozen[panel]["sha256"]
            for panel in paths
        ):
            raise ValueError(f"{stem}: scored prediction bytes differ from seal")
        typed, by_type = _typed(
            reports["typed"], model["model_id"], model["weight_revision"]
        )
        css, by_task = _css(reports["css"])
        if (
            reports["typed"].get("gold_sha256")
            != roster["panel"]["typed_final_gold_sha256"]
            or reports["css"].get("gold_sha256")
            != roster["panel"]["css_evaluation_gold_sha256"]
        ):
            raise ValueError(f"{stem}: scored gold bytes differ from roster")
        public = reports["public"]
        if (
            public.get("items") != 231
            or public.get("model_id") != model["model_id"]
            or public.get("model_revision") != model["weight_revision"]
            or public.get("panel_manifest_sha256")
            != roster["panel"]["jevbench_public_manifest_sha256"]
            or public.get("prompts_sha256")
            != roster["panel"]["jevbench_public_prompts_sha256"]
        ):
            raise ValueError(f"{stem}: public231 score differs from roster")
        rows.append(
            {
                "name": model["name"],
                "model_id": model["model_id"],
                "weight_revision": model["weight_revision"],
                "loaded_parameters": model["loaded_parameters"],
                "T": typed,
                "H": css,
                "score": 100 * math.sqrt(typed * css),
                "H_without_FLUTE": statistics.median(
                    value for task, value in by_task.items() if task != "flute"
                ),
                "typed_by_type": by_type,
                "css_by_task": by_task,
                "public231_accuracy_all": public["accuracy_all"],
                "public231_by_tier": {
                    tier: public["tiers"][tier]["accuracy_all"]
                    for tier in ("easy", "standard", "hard")
                },
                "public231_valid": public["valid"],
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["loaded_parameters"], row["name"]))
    for index, row in enumerate(rows):
        row["rank"] = (
            rows[index - 1]["rank"]
            if index and row["score"] == rows[index - 1]["score"]
            else index + 1
        )
        row["pareto_on_this_roster"] = not any(
            other is not row
            and other["loaded_parameters"] <= row["loaded_parameters"]
            and other["score"] >= row["score"]
            and (
                other["loaded_parameters"] < row["loaded_parameters"]
                or other["score"] > row["score"]
            )
            for other in rows
        )
    return {
        "schema": SCHEMA,
        "scope": "three-model, prospectively sealed post-key same-panel comparison",
        "not_never_unsealed_blind": True,
        "roster_sha256": file_digest(roster_path),
        "prediction_seal_sha256": file_digest(seal_path),
        "report_sha256": report_hashes,
        "formula": roster["scoring_lock"]["primary_score"],
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = rank(args.roster, args.seal, args.reports_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "rank_sha256": file_digest(args.output),
                "roster_sha256": result["roster_sha256"],
                "model_count": len(result["models"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
