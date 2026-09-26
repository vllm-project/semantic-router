"""Read-only audit of a completed locked final evaluation, without gold access.

This verifier reads the saved command plan, raw prediction bytes, v2 reports,
paired comparisons, and optional publication artifacts. It never opens a
synthetic final or CSS gold file and performs no inference or API call.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from publication.load import (
    benchmark_report,
    comparison_report,
    css_report,
    load_inputs,
)
from scripts.plan_final_eval import (
    CSS_EVALUATION_ITEMS,
    EIKOS_ARCHITECTURE,
    EVALUATION_TASKS,
    PINNED_PROTOCOL_SHA256,
    PLAN_VERSION,
    sha_file,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256sum(path: Path) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: malformed SHA-256 line")
        item = Path(match[2])
        if item in entries:
            raise ValueError(f"{path}:{line_number}: duplicate raw artifact")
        entries[item] = match[1]
    if not entries:
        raise ValueError(f"{path}: empty raw hash manifest")
    return entries


def _prediction_rows(path: Path, model: dict[str, Any], expected: int) -> None:
    seen: set[str] = set()
    private = {"gold", "provenance", "family", "split", "target_probs", "teacher_probs"}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or row["id"] in seen
            ):
                raise ValueError(
                    f"{path.name}:{line_number}: missing or duplicate prediction ID"
                )
            seen.add(row["id"])
            if (
                private & set(row)
                or not isinstance(row.get("answers"), dict)
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("source_input_sha256"))
                )
            ):
                raise ValueError(
                    f"{path.name}:{line_number}: private field or malformed prediction"
                )
            if model["key"] == "jev":
                if row.get("model") != model["revision"]:
                    raise ValueError(
                        f"{path.name}:{line_number}: Jev response model differs"
                    )
            elif model["key"].startswith("d2-"):
                frozen = model.get("frozen_model_sha256")
                if frozen is not None and (
                    row.get("model_sha256") != frozen
                    or row.get("calibration_sha256")
                    != model["frozen_calibration_sha256"]
                ):
                    raise ValueError(
                        f"{path.name}:{line_number}: Decision 2.0 model digest differs"
                    )
                if model.get("frozen_architecture") == EIKOS_ARCHITECTURE and (
                    row.get("model_id") != model["model_id"]
                    or row.get("model_revision") != model["revision"]
                    or row.get("backend") != "eikos-semif-native"
                    or row.get("adapter_version") != "decision2-eikos-semif-native-v1"
                ):
                    raise ValueError(
                        f"{path.name}:{line_number}: Eikos package identity differs"
                    )
            elif (
                row.get("backend") != model["backend"]
                or row.get("model_revision") != model["revision"]
                or row.get("revision_attested") is False
                or (
                    model["key"] in {"eos", "kai", "lex", "lux", "nox", "sol"}
                    and row.get("runtime_matches_validated") is not True
                )
            ):
                raise ValueError(
                    f"{path.name}:{line_number}: native model/runtime identity differs"
                )
    if len(seen) != expected:
        raise ValueError(
            f"{path.name}: expected {expected} complete prediction IDs, found {len(seen)}"
        )


def _ci95(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            type(bound) in (int, float) and math.isfinite(bound) and -1 <= bound <= 1
            for bound in value
        )
        and value[0] <= value[1]
    )


def audit(
    plan_path: Path, *, expected_plan_sha256: str, complete: bool = False
) -> dict[str, Any]:
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256) is None
        or sha_file(plan_path) != expected_plan_sha256
    ):
        raise ValueError(
            "Saved command plan differs from the pre-test recorded SHA-256"
        )
    plan = _json(plan_path)
    if (
        plan.get("plan_version") != PLAN_VERSION
        or plan.get("status") != "commands_only_not_executed"
    ):
        raise ValueError("Unknown locked final evaluation plan")
    source_root = Path(plan["source_root"])
    if (
        sha_file(source_root / "scripts/plan_final_eval.py")
        != plan["planner_source_sha256"]
    ):
        raise ValueError("Final evaluation planner changed after freeze")
    if sha_file(Path(__file__)) != plan["auditor_source_sha256"]:
        raise ValueError("Final evaluation auditor changed after freeze")
    for name, digest in PINNED_PROTOCOL_SHA256.items():
        if (
            plan["protocol_source_sha256"].get(name) != digest
            or sha_file(source_root / name) != digest
        ):
            raise ValueError(f"Protocol source changed after freeze: {name}")
    if sha_file(Path(plan["freeze_manifest_path"])) != plan["freeze_manifest_sha256"]:
        raise ValueError("Pre-test freeze declaration changed after planning")
    for candidate in plan["frozen_candidates"]:
        if candidate.get("architecture") == EIKOS_ARCHITECTURE:
            reports = candidate.get("parity_reports", {})
            hashes = candidate.get("parity_report_sha256", {})
            if (
                set(reports) != {"dev", "css_pilot"}
                or set(hashes) != set(reports)
                or any(
                    sha_file(Path(reports[panel])) != hashes[panel] for panel in reports
                )
            ):
                raise ValueError(
                    "Eikos pre-test package parity reports changed after planning"
                )
    css_prompt = plan["css_evaluation_prompts"]
    if sha_file(Path(css_prompt["path"])) != css_prompt["sha256"]:
        raise ValueError("Frozen CSS gold-free prompt bytes changed")
    raw = _sha256sum(Path(plan["raw_prediction_hashes_path"]))
    models = plan["inference"]
    if {item["key"] for item in plan["frozen_candidates"]} != {
        item["key"] for item in models if item["key"].startswith("d2-")
    }:
        raise ValueError(
            "Frozen Decision 2.0 candidates differ from planned inference models"
        )
    reports_by_key: dict[str, dict[str, Any]] = {}
    expected_raw: set[Path] = set()
    for model in models:
        expected_raw.update(Path(path) for path in model["predictions"].values())
        expected_raw.update(Path(path) for path in model["receipts"].values())
        expected_raw.update(Path(path) for path in model.get("manifests", {}).values())
    if set(raw) != expected_raw:
        raise ValueError(
            "Raw SHA-256 inventory differs from predeclared predictions/receipts"
        )
    for path, digest in raw.items():
        if sha_file(path) != digest:
            raise ValueError(
                f"Raw prediction or receipt changed after hash freeze: {path.name}"
            )
    score_paths = {entry["key"]: entry["reports"] for entry in plan["scoring"]}
    final_gold_hashes: set[str] = set()
    css_gold_hashes: set[str] = set()
    report_digests = {}
    for model in models:
        key = model["key"]
        if key not in score_paths:
            raise ValueError(f"{key}: missing planned score paths")
        prediction_sha = {
            panel: raw[Path(path)] for panel, path in model["predictions"].items()
        }
        frozen = next(
            (item for item in plan["frozen_candidates"] if item["key"] == key), None
        )
        if frozen is not None:
            model = {
                **model,
                "frozen_model_sha256": frozen["model_sha256"],
                "frozen_calibration_sha256": frozen["calibration_sha256"],
                "frozen_architecture": frozen.get("architecture", "qwen_dynamic"),
            }
        _prediction_rows(Path(model["predictions"]["final"]), model, 1600)
        _prediction_rows(
            Path(model["predictions"]["css-evaluation"]), model, CSS_EVALUATION_ITEMS
        )
        benchmark_path = Path(score_paths[key]["final"])
        css_path = Path(score_paths[key]["css-evaluation"])
        bench, css = _json(benchmark_path), _json(css_path)
        if (
            bench.get("schema_version") != "typed-decision-report/2"
            or bench.get("split") != "final"
            or bench.get("model")
            != {
                name: model[source]
                for name, source in (
                    ("id", "model_id"),
                    ("revision", "revision"),
                    ("backend", "backend"),
                )
            }
            or bench.get("predictions_sha256") != prediction_sha["final"]
            or bench.get("items") != 1600
            or bench.get("predicted_items") != 1600
            or set(bench.get("by_family", {}))
            != {
                "constraint_competition",
                "exception_stack",
                "evidence_join",
                "resource_ledger",
            }
        ):
            raise ValueError(
                f"{key}: incomplete or mismatched v2 synthetic final report"
            )
        if (
            css.get("score_schema_version") != "css-transfer-score/2"
            or css.get("panel_version") != "css-transfer/1"
            or css.get("predictions_sha256") != prediction_sha["css-evaluation"]
            or set(css.get("tasks", {})) != set(EVALUATION_TASKS)
            or css.get("roles", {}).get("evaluation", {}).get("items")
            != CSS_EVALUATION_ITEMS
        ):
            raise ValueError(
                f"{key}: incomplete or mismatched v2 CSS evaluation report"
            )
        benchmark_report(benchmark_path)
        css_report(css_path)
        final_gold_hashes.add(bench["gold_sha256"])
        css_gold_hashes.add(css["gold_sha256"])
        reports_by_key[key] = {"benchmark": bench, "css": css}
        report_digests[key] = {
            "final": sha_file(benchmark_path),
            "css_evaluation": sha_file(css_path),
        }
        if frozen is not None:
            for panel, pred_path in model["predictions"].items():
                manifest = _json(Path(pred_path + ".manifest.json"))
                cal = manifest.get("calibration", {})
                expected_adapter = (
                    "decision2-eikos-semif-native-v1"
                    if frozen.get("architecture") == EIKOS_ARCHITECTURE
                    else "decision2-typed-benchmark-adapter-v2-calibrated"
                )
                if (
                    manifest.get("adapter_version") != expected_adapter
                    or manifest.get("model_sha256") != frozen["model_sha256"]
                    or manifest.get("model_id") != frozen["model_id"]
                    or manifest.get("model_revision") != frozen["selected_checkpoint"]
                    or manifest.get("predictions_sha256") != prediction_sha[panel]
                    or cal.get("file_sha256") != frozen["calibration_sha256"]
                    or manifest.get("max_length") != frozen["max_length"]
                ):
                    raise ValueError(
                        f"{key}/{panel}: 2.0 prediction manifest differs from pre-test freeze"
                    )
                if frozen.get("architecture") == EIKOS_ARCHITECTURE and (
                    manifest.get("package_manifest_sha256") != frozen["model_sha256"]
                    or manifest.get("calibration_sha256")
                    != frozen["calibration_sha256"]
                    or manifest.get("package_files_checked", 0) < 1
                    or manifest.get("input_items")
                    != (1600 if panel == "final" else CSS_EVALUATION_ITEMS)
                    or manifest.get("evaluated_items") != manifest.get("input_items")
                ):
                    raise ValueError(
                        f"{key}/{panel}: Eikos package manifest differs from pre-test freeze"
                    )
    if len(final_gold_hashes) != 1 or len(css_gold_hashes) != 1:
        raise ValueError(
            "Models were not scored against identical final/CSS gold digests"
        )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (*final_gold_hashes, *css_gold_hashes)
    ):
        raise ValueError("A score report has a malformed gold digest")
    result = {
        "audit_version": "decision2-final-evaluation-audit/1",
        "plan_sha256": expected_plan_sha256,
        "raw_hash_manifest_sha256": sha_file(Path(plan["raw_prediction_hashes_path"])),
        "models": len(models),
        "v2_reports": len(models) * 2,
        "synthetic_final_gold_sha256_from_reports": next(iter(final_gold_hashes)),
        "css_evaluation_gold_sha256_from_reports": next(iter(css_gold_hashes)),
        "report_sha256": report_digests,
        "stage": "scores",
    }
    if not complete:
        return result

    for pair in plan["paired_ci"]:
        first, second = pair["candidate"], pair["baseline"]
        benchmark = _json(Path(pair["reports"]["final"]))
        left_score, right_score = (
            reports_by_key[first]["benchmark"],
            reports_by_key[second]["benchmark"],
        )
        if (
            benchmark.get("schema_version") != "typed-decision-comparison/1"
            or benchmark.get("split") != "final"
            or benchmark.get("models")
            != {
                "left": next(model for model in models if model["key"] == first)[
                    "model_id"
                ],
                "right": next(model for model in models if model["key"] == second)[
                    "model_id"
                ],
            }
            or benchmark.get("gold_sha256")
            != result["synthetic_final_gold_sha256_from_reports"]
            or benchmark.get("left_sha256")
            != raw[
                Path(
                    next(model for model in models if model["key"] == first)[
                        "predictions"
                    ]["final"]
                )
            ]
            or benchmark.get("right_sha256")
            != raw[
                Path(
                    next(model for model in models if model["key"] == second)[
                        "predictions"
                    ]["final"]
                )
            ]
            or benchmark.get("iterations") != 5000
            or benchmark.get("seed") != 20260926
            or benchmark.get("groups_by_family")
            != dict.fromkeys(
                (
                    "constraint_competition",
                    "exception_stack",
                    "evidence_join",
                    "resource_ledger",
                ),
                100,
            )
            or not _ci95(benchmark.get("family_macro", {}).get("delta_ci95"))
            or not math.isclose(
                benchmark.get("family_macro", {}).get("left", -1),
                left_score["macro_family_accuracy"],
                rel_tol=0,
                abs_tol=1e-10,
            )
            or not math.isclose(
                benchmark.get("family_macro", {}).get("right", -1),
                right_score["macro_family_accuracy"],
                rel_tol=0,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(f"{pair['key']}: mismatched synthetic paired CI")
        if set(benchmark.get("by_family", {})) != set(left_score["by_family"]) or any(
            not _ci95(benchmark["by_family"][name].get("delta_ci95"))
            or not math.isclose(
                benchmark["by_family"][name].get("left", -1),
                left_score["by_family"][name]["accuracy_all"],
                rel_tol=0,
                abs_tol=1e-10,
            )
            or not math.isclose(
                benchmark["by_family"][name].get("right", -1),
                right_score["by_family"][name]["accuracy_all"],
                rel_tol=0,
                abs_tol=1e-10,
            )
            for name in left_score["by_family"]
        ):
            raise ValueError(
                f"{pair['key']}: synthetic family CI differs from v2 score reports"
            )
        css_comparison = comparison_report(
            Path(pair["reports"]["css-evaluation"]),
            reports_by_key[first]["css"],
            reports_by_key[second]["css"],
        )
        if (
            css_comparison.get("model_a")
            != next(model for model in models if model["key"] == first)["model_id"]
            or css_comparison.get("model_b")
            != next(model for model in models if model["key"] == second)["model_id"]
            or css_comparison.get("bootstrap", {}).get("replicates") != 5000
            or css_comparison.get("bootstrap", {}).get("seed") != 20260926
            or css_comparison.get("compare_code_sha256")
            != PINNED_PROTOCOL_SHA256["transfer/compare.py"]
            or css_comparison.get("score_code_sha256")
            != PINNED_PROTOCOL_SHA256["transfer/score.py"]
        ):
            raise ValueError(
                f"{pair['key']}: CSS paired CI differs from the pinned protocol"
            )

    config_path = Path(plan["publication_config_path"])
    if _json(config_path) != plan["publication_config"]:
        raise ValueError("Publication config differs from predeclared model/pair set")
    load_inputs(
        config_path
    )  # Validates all score identities, gold, and CSS paired CI bindings.
    artifact_dir = Path(plan["card_artifacts_path"])
    manifest = _json(artifact_dir / "manifest.json")
    if (
        manifest.get("publication_version") != "decision-model-card-artifacts/2"
        or manifest.get("config_sha256") != sha_file(config_path)
        or manifest.get("generator_code_sha256")
        != {
            name: PINNED_PROTOCOL_SHA256[f"publication/{name}"]
            for name in ("generate.py", "load.py", "render.py")
        }
        or manifest.get("frozen_benchmark", {}).get("gold_sha256")
        != result["synthetic_final_gold_sha256_from_reports"]
        or manifest.get("css_gold_sha256")
        != result["css_evaluation_gold_sha256_from_reports"]
    ):
        raise ValueError("Rank/matrix publication manifest differs from frozen reports")
    artifact_models = manifest.get("models")
    if not isinstance(artifact_models, list) or {
        entry.get("key") for entry in artifact_models if isinstance(entry, dict)
    } != set(reports_by_key):
        raise ValueError("Rank/matrix manifest model set differs from the frozen plan")
    for entry in artifact_models:
        key = entry["key"]
        if (
            entry.get("benchmark", {}).get("report_sha256")
            != report_digests[key]["final"]
            or entry.get("css", {}).get("report_sha256")
            != report_digests[key]["css_evaluation"]
        ):
            raise ValueError(f"{key}: rank/matrix source report hash differs")
    for filename in ("score-table.md", "ranking.svg", "matrix.svg"):
        if manifest.get("artifacts_sha256", {}).get(filename) != sha_file(
            artifact_dir / filename
        ):
            raise ValueError(f"Publication artifact hash differs: {filename}")
    result["stage"] = "complete"
    result["paired_comparisons"] = len(plan["paired_ci"]) * 2
    result["card_manifest_sha256"] = sha_file(artifact_dir / "manifest.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Saved JSON output from scripts.plan_final_eval",
    )
    parser.add_argument(
        "--expected-plan-sha256",
        required=True,
        help="SHA-256 recorded outside the plan before either final panel ran",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Also require paired CIs and rank/matrix artifacts",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.plan,
                expected_plan_sha256=args.expected_plan_sha256,
                complete=args.complete,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
