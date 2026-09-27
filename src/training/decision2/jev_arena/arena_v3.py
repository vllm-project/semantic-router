"""Score the prospective JevArena v3 sealed core from completed reports.

This deliberately separate protocol never reads FINAL gold, model outputs or
public benchmark scores. A pre-key freeze receipt binds the two sealed panels
and candidate prediction digests. Its timing and package/runtime identity still
need an independent release audit; a successful rank is not that audit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from benchmark.generate import FINAL_FAMILIES
from transfer.build import EVALUATION_TASKS, PANEL_VERSION, PILOT_TASKS

from jev_arena.arena import _load, _pareto, _score, _sha

ARENA_VERSION = "jevarena-ranking/3"
ROSTER_VERSION = "jevarena-v3-roster/1"
FREEZE_VERSION = "jevarena-v3-freeze/1"
TYPES = ("choice", "noul", "score")
AXES = ("typed", "transfer")
PANEL_HASH_KEYS = ("typed_gold_sha256", "css_gold_sha256")
PREDICTION_KEYS = ("typed", "css")
SCORER_SOURCE_PATHS = {
    "arena_v3": Path(__file__),
    "paired_v3": Path(__file__).with_name("compare_v3.py"),
    "typed": Path(__file__).resolve().parents[1] / "benchmark/score.py",
    "css": Path(__file__).resolve().parents[1] / "transfer/score.py",
}
ENTRY_FIELDS = {
    "key",
    "label",
    "group",
    "model_id",
    "revision",
    "size_b",
    "typed_report",
    "css_report",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{name}: expected a lowercase SHA-256 digest")
    return value


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected nonempty text")
    return value


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError(
            f"{name}: expected a {'positive' if positive else 'nonnegative'} count"
        )
    return value


def _metric(value: Any, name: str) -> float:
    return _score(value, name)


def _close(actual: Any, expected: float, name: str) -> float:
    value = _metric(actual, name)
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{name}: disagrees with component scores")
    return value


def _summary(summary: Any, name: str) -> tuple[int, int, int, float]:
    if not isinstance(summary, dict):
        raise ValueError(f"{name}: summary is missing")
    n = _count(summary.get("n"), f"{name}.n", positive=True)
    valid = _count(summary.get("valid_n"), f"{name}.valid_n")
    correct = _count(summary.get("correct_n"), f"{name}.correct_n")
    if not 0 <= correct <= valid <= n:
        raise ValueError(f"{name}: impossible correct/valid/item counts")
    if summary.get("invalid_or_missing_n") != n - valid:
        raise ValueError(f"{name}: missing/invalid answers are not in the denominator")
    accuracy = _close(summary.get("accuracy_all"), correct / n, f"{name}.accuracy_all")
    return n, valid, correct, accuracy


def _partition(
    summaries: Any, keys: set[str], name: str
) -> tuple[dict[str, float], tuple[int, int, int]]:
    if not isinstance(summaries, dict) or set(summaries) != keys:
        raise ValueError(f"{name}: incomplete frozen partition")
    parsed = {key: _summary(summaries[key], f"{name}.{key}") for key in sorted(keys)}
    return (
        {key: value[3] for key, value in parsed.items()},
        tuple(sum(value[index] for value in parsed.values()) for index in range(3)),
    )


def _typed(
    report: dict[str, Any], model_id: str, revision: str
) -> tuple[float, dict[str, float]]:
    if (
        report.get("schema_version") != "typed-decision-report/2"
        or report.get("split") != "final"
        or report.get("items") != 1600
    ):
        raise ValueError("Typed report must be the 1600-item FINAL panel")
    model = report.get("model")
    if not isinstance(model, dict) or (
        model.get("id") != model_id or model.get("revision") != revision
    ):
        raise ValueError("Typed report model identity differs from roster")
    overall = _summary(report.get("overall"), "typed.overall")
    family, family_counts = _partition(
        report.get("by_family"), set(FINAL_FAMILIES), "typed.by_family"
    )
    by_type, type_counts = _partition(
        report.get("by_type"), set(TYPES), "typed.by_type"
    )
    if overall[:3] != family_counts or overall[:3] != type_counts or overall[0] != 1600:
        raise ValueError("Typed overall, family and task-type counts disagree")
    return (
        _close(
            report.get("macro_family_accuracy"),
            statistics.mean(family.values()),
            "typed.macro_family_accuracy",
        ),
        by_type,
    )


def _css(report: dict[str, Any]) -> tuple[float, dict[str, float]]:
    if (
        report.get("score_schema_version") != "css-transfer-score/2"
        or report.get("panel_version") != PANEL_VERSION
    ):
        raise ValueError("CSS report version differs from the frozen panel")
    tasks = report.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("CSS report lacks task details")
    if set(tasks) - set(EVALUATION_TASKS) - set(PILOT_TASKS):
        raise ValueError("CSS report contains an unknown task")
    if any(
        not isinstance(task, dict)
        or task.get("role") != ("evaluation" if name in EVALUATION_TASKS else "pilot")
        for name, task in tasks.items()
    ):
        raise ValueError("CSS task has an invalid role")
    evaluation = {
        name: task
        for name, task in tasks.items()
        if isinstance(task, dict) and task.get("role") == "evaluation"
    }
    if set(evaluation) != set(EVALUATION_TASKS):
        raise ValueError("CSS evaluation must include exactly 15 frozen tasks")
    scores: dict[str, float] = {}
    items = valid_items = correct_items = 0
    for name, task in sorted(evaluation.items()):
        n, valid, correct, _ = _summary(task, f"css.{name}")
        scores[name] = _metric(task.get("macro_f1_all"), f"css.{name}.macro_f1_all")
        items += n
        valid_items += valid
        correct_items += correct
    roles = report.get("roles")
    role = roles.get("evaluation", {}) if isinstance(roles, dict) else {}
    if (
        items != 6547
        or role.get("tasks") != 15
        or role.get("items") != items
        or role.get("valid_items") != valid_items
    ):
        raise ValueError("CSS evaluation item/task counts are incomplete")
    _close(
        role.get("micro_accuracy_all"), correct_items / items, "css.micro_accuracy_all"
    )
    return (
        _close(
            role.get("median_task_macro_f1_all"),
            statistics.median(scores.values()),
            "css.median_task_macro_f1_all",
        ),
        scores,
    )


def _freeze(path: Path, manifest: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    if _sha(path) != _digest(manifest.get("freeze_sha256"), "freeze_sha256"):
        raise ValueError("Pre-key freeze receipt digest changed")
    freeze = _load(path)
    if (
        freeze.get("schema_version") != FREEZE_VERSION
        or freeze.get("status") != "prekey_frozen"
    ):
        raise ValueError("Missing pre-key v3 freeze receipt")
    scorer_sources = freeze.get("score_sources_sha256")
    if not isinstance(scorer_sources, dict) or set(scorer_sources) != set(
        SCORER_SOURCE_PATHS
    ):
        raise ValueError("JevArena v3 freeze lacks all scoring source digests")
    if any(
        scorer_sources[name] != _sha(path) for name, path in SCORER_SOURCE_PATHS.items()
    ):
        raise ValueError("JevArena v3 scoring source changed after freeze")
    _digest(freeze.get("protocol_sha256"), "protocol_sha256")
    _digest(freeze.get("candidate_lock_sha256"), "candidate_lock_sha256")
    panels = freeze.get("panels")
    if not isinstance(panels, dict) or set(panels) != set(PANEL_HASH_KEYS):
        raise ValueError("Freeze receipt lacks the two sealed panel identities")
    for name in PANEL_HASH_KEYS:
        _digest(panels[name], name)
    frozen_models = freeze.get("models")
    if not isinstance(frozen_models, dict) or set(frozen_models) != keys:
        raise ValueError("Freeze receipt candidate roster differs from ranking roster")
    for key, model in frozen_models.items():
        if not isinstance(model, dict) or set(model) != {
            "model_id",
            "revision",
            "native_model_sha256",
            "adapter_sha256",
            "calibration_sha256",
            "predictions_sha256",
        }:
            raise ValueError(f"{key}: incomplete frozen model identity")
        for name in ("model_id", "revision"):
            _name(model[name], f"{key}.{name}")
        for name in ("native_model_sha256", "calibration_sha256"):
            if model[name] is not None:
                _digest(model[name], f"{key}.{name}")
        _digest(model["adapter_sha256"], f"{key}.adapter_sha256")
        predictions = model["predictions_sha256"]
        if not isinstance(predictions, dict) or set(predictions) != set(
            PREDICTION_KEYS
        ):
            raise ValueError(f"{key}: incomplete frozen prediction digests")
        for name in PREDICTION_KEYS:
            _digest(predictions[name], f"{key}.{name}.predictions_sha256")
    return freeze


def rank(manifest_path: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if (
        manifest.get("schema_version") != ROSTER_VERSION
        or manifest.get("phase") != "release"
    ):
        raise ValueError("JevArena v3 accepts only its release roster schema")
    entries = manifest.get("models")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("JevArena v3 requires at least two same-panel models")
    keys = [entry.get("key") for entry in entries if isinstance(entry, dict)]
    if (
        len(keys) != len(entries)
        or any(not isinstance(key, str) or not key for key in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("Missing or duplicate JevArena v3 model key")
    freeze_path = Path(_name(manifest.get("freeze_receipt"), "freeze_receipt"))
    if not freeze_path.is_absolute():
        freeze_path = manifest_path.parent / freeze_path
    freeze = _freeze(freeze_path, manifest, set(keys))
    rows: list[dict[str, Any]] = []
    expected_hashes = freeze["panels"]
    for entry in entries:
        if set(entry) != ENTRY_FIELDS:
            raise ValueError("Incomplete JevArena v3 roster entry")
        key = entry["key"]
        frozen = freeze["models"][key]
        model_id = _name(entry["model_id"], f"{key}.model_id")
        revision = _name(entry["revision"], f"{key}.revision")
        if (model_id, revision) != (frozen["model_id"], frozen["revision"]):
            raise ValueError(f"{key}: frozen model identity mismatch")
        if entry["group"] == "decision2" and frozen["native_model_sha256"] is None:
            raise ValueError(f"{key}: Decision 2.0 requires a native model fingerprint")
        size = entry["size_b"]
        if size is not None and (
            type(size) not in (int, float) or not math.isfinite(size) or size <= 0
        ):
            raise ValueError(
                f"{key}: size_b must be positive measured parameters in billions or null"
            )
        paths = {}
        for name, field in (
            ("typed", "typed_report"),
            ("css", "css_report"),
        ):
            path = Path(_name(entry[field], f"{key}.{field}"))
            paths[name] = path if path.is_absolute() else manifest_path.parent / path
        typed, css = (_load(paths[name]) for name in PREDICTION_KEYS)
        panel_hashes = {
            "typed_gold_sha256": typed.get("gold_sha256"),
            "css_gold_sha256": css.get("gold_sha256"),
        }
        if panel_hashes != expected_hashes:
            raise ValueError(f"{key}: sealed panel digest differs from pre-key freeze")
        for name, report in (("typed", typed), ("css", css)):
            if report.get("predictions_sha256") != frozen["predictions_sha256"][name]:
                raise ValueError(
                    f"{key}: {name} predictions differ from pre-key freeze"
                )
        axes = {}
        axes["typed"], by_type = _typed(typed, model_id, revision)
        axes["transfer"], by_task = _css(css)
        score = 100 * math.prod(axes.values()) ** (1 / len(AXES))
        rows.append(
            {
                "key": key,
                "label": _name(entry["label"], f"{key}.label"),
                "group": _name(entry["group"], f"{key}.group"),
                "model_id": model_id,
                "revision": revision,
                "size_b": float(size) if size is not None else None,
                "native_model_sha256": frozen["native_model_sha256"],
                "adapter_sha256": frozen["adapter_sha256"],
                "calibration_sha256": frozen["calibration_sha256"],
                "axes": axes,
                "score": score,
                "task_scores": {
                    "typed": by_type,
                    "transfer": by_task,
                },
                "coverage": {
                    "typed_items": 1600,
                    "css_items": 6547,
                    "sealed_core_items": 8147,
                },
                "report_sha256": {name: _sha(path) for name, path in paths.items()},
            }
        )
    rows.sort(
        key=lambda row: (
            -row["score"],
            -row["axes"]["transfer"],
            row["key"],
        )
    )
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    _pareto(rows)
    return {
        "schema_version": ARENA_VERSION,
        "phase": "release",
        "status": "scored_pending_independent_release_audit",
        "manifest_sha256": _sha(manifest_path),
        "freeze_sha256": _sha(freeze_path),
        "panel_sha256": expected_hashes,
        "policy": {
            "score": "100 times the geometric mean of typed and transfer fractions.",
            "axes": list(AXES),
            "public_benchmarks": "JevBench and Decision Bench are excluded from this sealed-core score.",
            "invalid": "Missing and invalid answers remain in every report denominator.",
            "release_audit": "This scorer checks digest consistency, not independent pre-key timing or package/runtime parity.",
            "tie": "Score descending, transfer descending, key ascending.",
        },
        "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = rank(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "models": len(report["models"]),
                "top": report["models"][0]["key"],
                "status": report["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
