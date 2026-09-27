"""Audit fixed 0.8B CAL temperature transport without model inference.

This development-only tool verifies existing receipts, reconstructs T=1 from
strictly positive calibrated probabilities, and scores the same frozen panels.
It never fits a new temperature or selects a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
from typing import Any

from benchmark.score import score_suite
from research.calibration_transport import transform
from training.model.calibration import validate_temperatures
from transfer.score import score as score_css


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path.name}: SHA-256 mismatch: {actual}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def check_source(
    predictions: Path,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    calibration_sha: str,
    expected: dict[str, Any],
    panel: dict[str, Any],
) -> None:
    """Fail before scoring if any row or native sidecar breaks identity/positivity."""
    if (
        manifest.get("model_sha256") != expected["model_sha256"]
        or manifest.get("model_revision") != expected["checkpoint"]
        or manifest.get("predictions_sha256") != panel["predictions_sha256"]
        or manifest.get("calibration", {}).get("file_sha256") != calibration_sha
        or manifest.get("calibration", {}).get("temperature_by_type")
        != calibration.get("temperature_by_type")
    ):
        raise ValueError("native prediction sidecar does not bind frozen model/CAL")
    counts = manifest.get("counts", {})
    if (
        counts.get("items") != panel["n"]
        or counts.get("questions") != panel["n"]
        or counts.get("valid_questions") != panel["n"]
        or counts.get("invalid_questions") != 0
        or counts.get("over_budget_questions") != 0
        or counts.get("truncated_questions") != 0
    ):
        raise ValueError("native sidecar is not a complete all-valid panel")
    seen: set[str] = set()
    questions = 0
    with predictions.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            item_id = row.get("id")
            if not isinstance(item_id, str) or not item_id or item_id in seen:
                raise ValueError("prediction IDs must be unique nonempty strings")
            seen.add(item_id)
            if (
                row.get("model_sha256") != expected["model_sha256"]
                or row.get("calibration_sha256") != calibration_sha
                or row.get("adapter_sha256") != manifest.get("adapter_sha256")
                or row.get("adapter_status") != "ok"
                or row.get("truncated_questions") != 0
            ):
                raise ValueError("prediction row identity, status or length differs")
            answers = row.get("answers")
            if not isinstance(answers, dict) or len(answers) != 1:
                raise ValueError("expected one native answer per panel row")
            answer = next(iter(answers.values()))
            kind = answer.get("type")
            if kind not in ("choice", "noul", "score"):
                raise ValueError("unexpected native answer type")
            if kind == "noul":
                p = answer.get("noul")
                values = (p, 1 - p) if type(p) in (int, float) else ()
            else:
                probabilities = answer.get("probabilities")
                values = (
                    tuple(probabilities.values())
                    if isinstance(probabilities, dict) and len(probabilities) >= 2
                    else ()
                )
            if not values or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value < 1
                for value in values
            ):
                raise ValueError(
                    "zero/one/nonfinite native probability blocks inversion"
                )
            if abs(sum(values) - 1.0) > 1e-6:
                raise ValueError("native probabilities do not sum to one")
            questions += 1
    if len(seen) != panel["n"] or questions != panel["n"]:
        raise ValueError("prediction count differs from frozen panel")


def point_answer(answer: dict[str, Any]) -> str | bool | int | None:
    kind = answer["type"]
    if kind == "choice":
        return answer["choice"]
    if kind == "noul":
        value = answer["noul"]
        return None if value == 0.5 else value > 0.5
    probabilities = answer["probabilities"]
    highest = max(probabilities.values())
    winners = [
        label for label, value in probabilities.items() if abs(value - highest) <= 1e-8
    ]
    return int(winners[0]) if len(winners) == 1 else None


def check_transformed(original: Path, transformed: Path, count: int) -> None:
    """Positive scaling must preserve each answer, including Score argmax."""
    observed = 0
    with original.open(encoding="utf-8") as source, transformed.open(
        encoding="utf-8"
    ) as derived:
        for source_line, derived_line in zip_longest(source, derived):
            if source_line is None or derived_line is None:
                raise ValueError("T=1 diagnostic row count differs from source")
            before, after = json.loads(source_line), json.loads(derived_line)
            if before["id"] != after["id"] or set(before["answers"]) != set(
                after["answers"]
            ):
                raise ValueError("T=1 diagnostic row or question order changed")
            for question in before["answers"]:
                left, right = before["answers"][question], after["answers"][question]
                if left["type"] != right["type"] or point_answer(left) != point_answer(
                    right
                ):
                    raise ValueError("T=1 diagnostic changed a categorical answer")
                if right["type"] == "noul":
                    values = (right["noul"], 1 - right["noul"])
                else:
                    values = tuple(right["probabilities"].values())
                if (
                    any(
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or not 0 < value < 1
                        for value in values
                    )
                    or abs(sum(values) - 1.0) > 1e-6
                ):
                    raise ValueError(
                        "T=1 diagnostic has a boundary/invalid probability"
                    )
                if right["type"] == "score" and not math.isclose(
                    right["score"],
                    sum(
                        int(level) * probability
                        for level, probability in right["probabilities"].items()
                    ),
                    abs_tol=1e-8,
                ):
                    raise ValueError("T=1 Score expectation differs from its map")
            observed += 1
    if observed != count:
        raise ValueError("T=1 diagnostic has the wrong item count")


def almost_equal(left: Any, right: Any) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(left, right, abs_tol=1e-10, rel_tol=1e-10)
    return left == right


def verify_old_score(actual: dict[str, Any], saved: dict[str, Any], panel: str) -> None:
    if panel == "dev":
        fields = ("n", "valid_n", "correct_n", "brier", "nll", "ece_10", "score_mae")
        for name in fields:
            if not almost_equal(
                actual["overall"].get(name), saved["overall"].get(name)
            ):
                raise ValueError(f"typed scorer drifted from the original {name}")
        for kind in ("choice", "noul", "score"):
            for name in fields:
                if not almost_equal(
                    actual["by_type"][kind].get(name), saved["by_type"][kind].get(name)
                ):
                    raise ValueError(f"typed scorer drifted for {kind}/{name}")
    else:
        fields = (
            "n",
            "valid_n",
            "correct_n",
            "macro_f1_all",
            "brier_sum",
            "nll",
            "ece_pmax_15",
        )
        if set(actual["tasks"]) != set(saved["tasks"]):
            raise ValueError("CSS task roster changed")
        for task in saved["tasks"]:
            for name in fields:
                if not almost_equal(
                    actual["tasks"][task].get(name), saved["tasks"][task].get(name)
                ):
                    raise ValueError(f"CSS scorer drifted for {task}/{name}")


def summarize_typed(report: dict[str, Any]) -> dict[str, Any]:
    fields = ("n", "valid_n", "correct_n", "brier", "nll", "ece_10", "score_mae")
    return {
        "overall": {name: report["overall"].get(name) for name in fields},
        "by_type": {
            kind: {name: report["by_type"][kind].get(name) for name in fields}
            for kind in ("choice", "noul", "score")
        },
    }


def summarize_css(report: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "n",
        "valid_n",
        "correct_n",
        "macro_f1_all",
        "brier_sum",
        "nll",
        "ece_pmax_15",
    )
    return {
        "pilot": {
            name: report["roles"]["pilot"].get(name)
            for name in (
                "items",
                "valid_items",
                "micro_accuracy_all",
                "median_task_macro_f1_all",
                "median_task_brier_sum",
                "median_task_ece_pmax_15",
            )
        },
        "tasks": {
            task: {name: data.get(name) for name in fields}
            for task, data in sorted(report["tasks"].items())
        },
    }


def check_invariance(before: dict[str, Any], after: dict[str, Any], panel: str) -> None:
    if panel == "dev":
        slices = [(before["overall"], after["overall"])] + [
            (before["by_type"][kind], after["by_type"][kind])
            for kind in ("choice", "noul", "score")
        ]
        fields = ("n", "valid_n", "correct_n")
    else:
        slices = [
            (before["tasks"][task], after["tasks"][task]) for task in before["tasks"]
        ]
        fields = ("n", "valid_n", "correct_n", "macro_f1_all")
    for left, right in slices:
        if any(not almost_equal(left[name], right[name]) for name in fields):
            raise ValueError("positive-temperature inversion changed point decisions")


def audit(
    evidence_path: Path,
    run_name: str,
    run_dir: Path,
    gold_dev: Path,
    gold_css: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = load_json(evidence_path)
    if evidence.get("version") != "decision2-eos08-temperature-transport/1":
        raise ValueError("wrong frozen evidence version")
    expected = evidence["runs"][run_name]
    gold = {"dev": gold_dev, "css": gold_css}
    for name, path in gold.items():
        require_hash(path, evidence["gold_sha256"][name])
    cal_path = run_dir / expected["calibration"]
    require_hash(cal_path, expected["calibration_sha256"])
    calibration = load_json(cal_path)
    if (
        calibration.get("model_sha256") != expected["model_sha256"]
        or calibration.get("selected_checkpoint") != expected["checkpoint"]
        or calibration.get("fit_split") != "cal"
        or calibration.get("overall", {}).get("after", {}).get("n")
        != expected["cal_rows"]
    ):
        raise ValueError("CAL is not the frozen selected model/partition")
    validate_temperatures(calibration.get("temperature_by_type"))
    sources = {}
    for panel_name, panel in expected["panels"].items():
        predictions = run_dir / panel["predictions"]
        sidecar = predictions.with_name(predictions.name + ".manifest.json")
        old_report = run_dir / panel["score"]
        require_hash(predictions, panel["predictions_sha256"])
        require_hash(sidecar, panel["manifest_sha256"])
        require_hash(old_report, panel["score_sha256"])
        manifest = load_json(sidecar)
        check_source(
            predictions,
            manifest,
            calibration,
            expected["calibration_sha256"],
            expected,
            panel,
        )
        saved = load_json(old_report)
        if (
            saved.get("gold_sha256") != evidence["gold_sha256"][panel_name]
            or saved.get("predictions_sha256") != panel["predictions_sha256"]
        ):
            raise ValueError("saved panel score identity differs from frozen inputs")
        if panel_name == "dev":
            identity = saved["model"]
            before = score_suite(
                gold[panel_name],
                predictions,
                identity["id"],
                identity["revision"],
                identity["backend"],
            )
        else:
            before = score_css(gold[panel_name], predictions)
        verify_old_score(before, saved, panel_name)
        sources[panel_name] = (predictions, before)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    summaries = {}
    for panel_name, (predictions, before) in sources.items():
        transformed = output_dir / f"{panel_name}.t1.diagnostic.jsonl"
        receipt = transform(predictions, cal_path, transformed)
        if receipt["items"] != expected["panels"][panel_name]["n"]:
            raise ValueError("temperature inversion dropped rows")
        check_transformed(predictions, transformed, receipt["items"])
        if panel_name == "dev":
            identity = before["model"]
            after = score_suite(
                gold[panel_name],
                transformed,
                identity["id"],
                identity["revision"],
                "T1-diagnostic-only",
            )
            select = summarize_typed
        else:
            after = score_css(gold[panel_name], transformed)
            select = summarize_css
        check_invariance(before, after, panel_name)
        (output_dir / f"{panel_name}.t1.score.private.json").write_text(
            json.dumps(after, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries[panel_name] = {
            "source_predictions_sha256": sha256(predictions),
            "t1_predictions_sha256": sha256(transformed),
            "before_calibrated": select(before),
            "after_t1": select(after),
        }
    result = {
        "diagnostic_only": True,
        "run_name": run_name,
        "model_sha256": expected["model_sha256"],
        "selected_checkpoint": expected["checkpoint"],
        "calibration_sha256": expected["calibration_sha256"],
        "temperature_by_type": calibration["temperature_by_type"],
        "cal_fitted_metrics": calibration["overall"],
        "evidence_sha256": sha256(evidence_path),
        "script_sha256": sha256(Path(__file__)),
        "transform_sha256": sha256(Path(transform.__code__.co_filename)),
        "typed_scorer_sha256": sha256(Path(score_suite.__code__.co_filename)),
        "css_scorer_sha256": sha256(Path(score_css.__code__.co_filename)),
        "panels": summaries,
    }
    (output_dir / "summary.private.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold-dev", type=Path, required=True)
    parser.add_argument("--gold-css", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.evidence,
        args.run_name,
        args.run_dir,
        args.gold_dev,
        args.gold_css,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "run_name": report["run_name"],
                "model_sha256": report["model_sha256"],
                "output_summary_sha256": sha256(
                    args.output_dir / "summary.private.json"
                ),
                "point_invariant": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
