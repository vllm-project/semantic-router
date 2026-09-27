"""Seal gold-free own-Sol 2B v3 predictions after two documented repairs.

The original frozen seal incorrectly sought ``adapter_version`` inside each
package prediction row; the collector puts it in the signed panel manifest.
The own-Sol-1 collector also aborted on four over-length CSS items. A separate
receipted continuation preserved every in-budget native answer and represented
those four as invalid. Neither repair changes a model, prompt, score or label.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference.run import ADAPTER_VERSION as BASELINE_ADAPTER_VERSION
from inference.run import digest as baseline_digest
from publication.adapter_bundle import _verify_staged_runtime
from publication.package_native_arena import ADAPTER_VERSION as PACKAGE_ADAPTER_VERSION
from publication.package_native_arena import input_digest, load_gold_free
from scripts import preflight_sol2b_v3 as frozen
from scripts.plan_final_eval import sha_file

SCHEMA = "decision2-sol2b-v3-postkey-prediction-seal/2"
AMENDMENT = "decision2-sol2b-own1-css-overbudget-amendment/1"


def _candidate_manifest(
    path: Path, prediction: Path, panel: dict[str, Any], answers: int
) -> dict[str, Any]:
    manifest = frozen._object(path)
    if (
        manifest.get("adapter_version") != PACKAGE_ADAPTER_VERSION
        or manifest.get("package_manifest_sha256") != frozen.PACKAGE_SHA
        or manifest.get("model_id") != frozen.CANDIDATE_ID
        or manifest.get("model_revision") != f"package-sha256:{frozen.PACKAGE_SHA}"
        or manifest.get("model_sha256") != frozen.CANDIDATE_SHA
        or manifest.get("calibration_sha256") != frozen.CAL_SHA
        or manifest.get("input_sha256") != panel["sha256"]
        or manifest.get("predictions_sha256") != sha_file(prediction)
        or manifest.get("input_items") != panel["items"]
        or manifest.get("counts", {}).get("items") != panel["items"]
        or manifest.get("counts", {}).get("questions") != answers
        or manifest.get("counts", {}).get("truncated_questions") != 0
    ):
        raise ValueError("Candidate native panel manifest changed")
    return manifest


def _own1_amendment(root: Path, full_css: Path) -> dict[str, Any]:
    plan = frozen._object(root / "own1.css.plan.json")
    stitch = frozen._object(root / "own1.css.stitch.json")
    partial = root / "own1.css.partial.predictions.jsonl"
    suffix = root / "own1.css.remaining-valid.predictions.jsonl"
    filtered = root / "own1.css.remaining-valid.prompts.jsonl"
    if (
        plan.get("schema_version") != AMENDMENT
        or stitch.get("schema_version") != AMENDMENT
        or plan.get("native_revision") != frozen.OWN_REVISION
        or plan.get("native_adapter_version") != BASELINE_ADAPTER_VERSION
        or plan.get("prompt_sha256") != frozen.PANELS["css"][2]
        or plan.get("prefix_sha256") != sha_file(partial)
        or plan.get("filtered_sha256") != sha_file(filtered)
        or stitch.get("plan_sha256") != sha_file(root / "own1.css.plan.json")
        or stitch.get("prefix_sha256") != sha_file(partial)
        or stitch.get("suffix_sha256") != sha_file(suffix)
        or stitch.get("full_predictions_sha256") != sha_file(full_css)
        or stitch.get("total_items") != frozen.PANELS["css"][0]
        or plan.get("over_budget_items") != stitch.get("over_budget_items")
    ):
        raise ValueError("Own-Sol-1 over-budget continuation changed")
    expected = {row["id"] for row in plan["over_budget"]}
    if len(expected) != len(plan["over_budget"]) or len(expected) != 4:
        raise ValueError("Own-Sol-1 over-budget count changed")
    return {
        "plan_sha256": sha_file(root / "own1.css.plan.json"),
        "stitch_sha256": sha_file(root / "own1.css.stitch.json"),
        "partial_sha256": sha_file(partial),
        "suffix_sha256": sha_file(suffix),
        "helper_sha256": sha_file(
            Path(__file__).with_name("collect_sol2b_own1_overbudget.py")
        ),
        "invalid_ids": expected,
    }


def seal(lock_path: Path, lock_sha: str, amendment_dir: Path, output: Path) -> str:
    if sha_file(lock_path) != lock_sha:
        raise ValueError("Original prospective candidate lock changed")
    lock = frozen._object(lock_path)
    if lock.get("schema_version") != frozen.LOCK_VERSION:
        raise ValueError("Unknown candidate lock")
    package = Path(lock["model"]["package"])
    source = Path(lock["comparators"]["own1"]["model_dir"])
    if sha_file(package / "MODEL_MANIFEST.json") != frozen.PACKAGE_SHA:
        raise ValueError("Candidate package changed")
    _verify_staged_runtime(package, source)
    source_root = Path(__file__).resolve().parents[1]
    if frozen._code(source_root) != lock["runtime"]["source_sha256"]:
        raise ValueError("Frozen inference and scoring source changed")
    prompts = {}
    for key, panel in lock["panels"].items():
        if frozen._panel(Path(panel["path"]), key) != panel:
            raise ValueError("Frozen panel changed")
        prompts[key] = load_gold_free(Path(panel["path"]))
    predictions = {}
    own1_css = Path(lock["prediction_dir"]) / "own1.css.predictions.jsonl"
    continuation = _own1_amendment(amendment_dir, own1_css)
    for model_key in ("candidate", "own1", "decider2b"):
        predictions[model_key] = {}
        for panel_name, prompt_rows in prompts.items():
            path = (
                Path(lock["prediction_dir"])
                / f"{model_key}.{panel_name}.predictions.jsonl"
            )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            if len(rows) != len(prompt_rows):
                raise ValueError("Missing formal prediction rows")
            manifest = (
                _candidate_manifest(
                    Path(str(path) + ".manifest.json"),
                    path,
                    lock["panels"][panel_name],
                    sum(len(row["questions"]) for row in prompt_rows),
                )
                if model_key == "candidate"
                else None
            )
            invalid_own1 = set()
            for prompt, row in zip(prompt_rows, rows):
                expected_input = (
                    input_digest(prompt)
                    if model_key == "candidate"
                    else baseline_digest(
                        {"state": prompt["state"], "questions": prompt["questions"]}
                    )
                )
                if (
                    row.get("id") != prompt["id"]
                    or row.get("source_input_sha256") != expected_input
                    or set(row.get("answers", {})) != set(prompt["questions"])
                ):
                    raise ValueError("Prediction does not match frozen prompt")
                if model_key == "candidate":
                    if (
                        row.get("model_id") != frozen.CANDIDATE_ID
                        or row.get("model_revision")
                        != f"package-sha256:{frozen.PACKAGE_SHA}"
                        or row.get("package_manifest_sha256") != frozen.PACKAGE_SHA
                        or row.get("model_sha256") != frozen.CANDIDATE_SHA
                        or row.get("calibration_sha256") != frozen.CAL_SHA
                        or row.get("adapter_sha256") != manifest["adapter_sha256"]
                        or row.get("input_sha256") != expected_input
                        or row.get("truncated_questions") != 0
                    ):
                        raise ValueError("Candidate row identity or truncation changed")
                else:
                    comparator = lock["comparators"][model_key]
                    if (
                        row.get("model_id")
                        != (frozen.OWN_ID if model_key == "own1" else None)
                        or row.get("model_revision") != comparator["revision"]
                        or row.get("backend")
                        != ("sol" if model_key == "own1" else "decider")
                        or row.get("adapter_version") != BASELINE_ADAPTER_VERSION
                        or row.get("revision_attested") is not True
                    ):
                        raise ValueError("Comparator row identity changed")
                    if (
                        model_key == "own1"
                        and panel_name == "css"
                        and row.get("invalid_reason")
                        == "native_max_length_exceeded_no_truncation"
                    ):
                        if row["answers"] != {"label": None}:
                            raise ValueError("Over-budget row was not invalid")
                        invalid_own1.add(row["id"])
            if model_key == "own1" and panel_name == "css":
                if invalid_own1 != continuation["invalid_ids"]:
                    raise ValueError("Own-Sol-1 invalid rows differ from token audit")
            predictions[model_key][panel_name] = {
                "items": len(rows),
                "answers": sum(len(row["answers"]) for row in rows),
                "predictions_sha256": sha_file(path),
                "native_manifest_sha256": (
                    sha_file(Path(str(path) + ".manifest.json"))
                    if manifest is not None
                    else None
                ),
            }
    del continuation["invalid_ids"]
    value = {
        "schema_version": SCHEMA,
        "status": "all_gold_free_predictions_sealed; no_scores_or_release_qualification",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "candidate_lock_sha256": lock_sha,
        "label_exposure": lock["label_exposure"],
        "amendment": {
            "reason": "frozen v1 checker expected row adapter_version; actual signed native manifest owns that field",
            "v1_seal_status": "failed_before_any_score_or_gold_read",
            "v2_sealer_sha256": sha_file(Path(__file__)),
            "own1_over_budget": continuation,
        },
        "predictions": predictions,
    }
    return frozen._write_private(output, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--own1-overbudget-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = seal(
        args.lock, args.expected_lock_sha256, args.own1_overbudget_dir, args.output
    )
    print(json.dumps({"schema_version": SCHEMA, "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
