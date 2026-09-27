"""Score the prospective JevArena v3 sealed core from completed reports.

This deliberately separate protocol never reads FINAL gold, model outputs or
public benchmark scores. A pre-key freeze receipt binds the two sealed panels
and candidate prediction digests. Its timing and package/runtime identity still
need an independent release audit; a successful rank is not that audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.generate import FINAL_FAMILIES
from transfer.build import EVALUATION_TASKS, PANEL_VERSION, PILOT_TASKS

from jev_arena.arena import _load, _pareto, _score, _sha

ARENA_VERSION = "jevarena-ranking/3"
ROSTER_VERSION = "jevarena-v3-roster/1"
FREEZE_VERSION = "jevarena-v3-freeze/2"
CHRONOLOGY_VERSION = "jevarena-v3-prekey-chronology/1"
TYPES = ("choice", "noul", "score")
AXES = ("typed", "transfer")
PANEL_HASH_KEYS = ("typed_gold_sha256", "css_gold_sha256")
SCORED_KEYS = ("typed", "css")
PREDICTION_KEYS = (*SCORED_KEYS, "public")
SCORER_SOURCE_PATHS = {
    "arena_v3": Path(__file__),
    "arena_base": Path(__file__).with_name("arena.py"),
    "paired_v3": Path(__file__).with_name("compare_v3.py"),
    "typed": Path(__file__).resolve().parents[1] / "benchmark/score.py",
    "typed_panel": Path(__file__).resolve().parents[1] / "benchmark/generate.py",
    "css": Path(__file__).resolve().parents[1] / "transfer/score.py",
    "css_panel": Path(__file__).resolve().parents[1] / "transfer/build.py",
    "css_compare": Path(__file__).resolve().parents[1] / "transfer/compare.py",
}
REQUIRED_PROTOCOL_SOURCES = {
    path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    for path in SCORER_SOURCE_PATHS.values()
} | {
    "publication/bundle_arena_v3.py",
    "scripts/plan_first_release_v3.py",
    "scripts/freeze_first_release_v3.py",
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


def _receipt(reference: Any, base: Path, name: str) -> dict[str, Any]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{name}: expected a path and SHA-256")
    path = Path(_name(reference["path"], f"{name}.path"))
    if not path.is_absolute():
        path = base / path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name}: receipt is missing or linked")
    if _sha(path) != _digest(reference["sha256"], f"{name}.sha256"):
        raise ValueError(f"{name}: receipt digest changed")
    return _load(path)


def _pair_digest(pairs: Any) -> str:
    if not isinstance(pairs, list) or any(
        not isinstance(pair, dict)
        or set(pair) != {"candidate", "comparator", "size_relation", "rationale"}
        or not isinstance(pair["candidate"], str)
        or not pair["candidate"]
        or not isinstance(pair["comparator"], str)
        or not pair["comparator"]
        or not isinstance(pair["size_relation"], str)
        or pair["size_relation"] not in {"same", "nearest"}
        or not isinstance(pair["rationale"], str)
        for pair in pairs
    ):
        raise ValueError("Freeze lacks predeclared comparison pairs")
    ordered = sorted(pairs, key=lambda pair: pair["candidate"])
    encoded = json.dumps(
        ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field}: invalid UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"{field}: timestamp must be UTC")
    return parsed


def _chronology(freeze: dict[str, Any], path: Path, frozen_at: datetime) -> None:
    """Bind declared event order; independent timestamp proof remains external."""
    chain = _receipt(freeze.get("chronology"), path.parent, "pre-key chronology")
    stages = ("candidate_lock", "prediction_seal", "audit_seal")
    if (
        set(chain) != {"schema_version", *stages}
        or chain["schema_version"] != CHRONOLOGY_VERSION
    ):
        raise ValueError("Pre-key chronology has an unknown or incomplete schema")
    expected = (
        freeze.get("candidate_lock_sha256"),
        freeze.get("raw_prediction_hashes_sha256"),
        freeze.get("prediction_audit", {}).get("sha256"),
    )
    previous: datetime | None = None
    for stage, digest in zip(stages, expected, strict=True):
        item = chain[stage]
        if not isinstance(item, dict) or set(item) != {"at_utc", "sha256"}:
            raise ValueError(f"{stage}: incomplete chronology event")
        if _digest(item["sha256"], f"{stage}.sha256") != digest:
            raise ValueError(f"{stage}: chronology digest differs from freeze")
        observed = _utc(item["at_utc"], f"{stage}.at_utc")
        if previous is not None and observed <= previous:
            raise ValueError("Pre-key chronology is not strictly ordered")
        previous = observed
    if previous is None or previous >= frozen_at:
        raise ValueError("Pre-key freeze must follow the gold-free audit seal")


def _prekey_evidence(
    freeze: dict[str, Any], path: Path, keys: set[str]
) -> dict[str, Any]:
    plan = _receipt(freeze.get("plan"), path.parent, "pre-key plan")
    audit = _receipt(freeze.get("prediction_audit"), path.parent, "gold-free audit")
    if plan.get("plan_version") != "decision2-first-release-v3-plan/2":
        raise ValueError("Pre-key freeze lacks the v3 first-release plan")
    if audit.get("status") != "gold_free_prekey_predictions_verified":
        raise ValueError("Pre-key freeze lacks a passed gold-free prediction audit")
    if plan.get("candidate_freeze_sha256") != freeze.get(
        "candidate_lock_sha256"
    ) or plan.get("gate_document_sha256") != freeze.get("protocol_sha256"):
        raise ValueError("Pre-key plan candidate lock or policy differs")
    if (
        plan.get("formula") != "100*sqrt(T*H)"
        or freeze.get("formula") != plan["formula"]
        or freeze.get("paired_bootstrap") != {"replicates": 5000, "seed": 20260927}
    ):
        raise ValueError("Pre-key formula or paired bootstrap policy differs")
    pairs = freeze.get("comparison_pairs")
    pair_sha = _digest(freeze.get("comparison_pairs_sha256"), "comparison_pairs_sha256")
    if (
        _pair_digest(pairs) != pair_sha
        or plan.get("comparison_pairs") != pairs
        or plan.get("comparison_pairs_sha256") != pair_sha
        or audit.get("comparison_pairs_sha256") != pair_sha
    ):
        raise ValueError("Predeclared comparison pairs differ from plan or audit")
    root = Path(_name(plan.get("source_root"), "plan.source_root")).resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("Pre-key plan points to another source checkout")
    source_hashes = plan.get("source_sha256")
    if not isinstance(source_hashes, dict) or not REQUIRED_PROTOCOL_SOURCES.issubset(
        source_hashes
    ):
        raise ValueError("Pre-key plan omits executable scoring or gate sources")
    for relative, digest in source_hashes.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or _sha(root / relative) != _digest(digest, f"source {relative}")
        ):
            raise ValueError(f"Pre-key protocol source changed: {relative}")
    roster = plan.get("model_roster")
    if (
        not isinstance(roster, list)
        or len(roster) != len(keys)
        or any(not isinstance(row, dict) for row in roster)
        or {row.get("key") for row in roster} != keys
    ):
        raise ValueError("Pre-key plan model roster differs from freeze")
    planned_groups = {row["key"]: row.get("group") for row in roster}
    candidate_keys = {
        key for key, group in planned_groups.items() if group == "decision2"
    }
    if (
        len(pairs) != len(candidate_keys)
        or {pair["candidate"] for pair in pairs} != candidate_keys
        or any(planned_groups.get(pair["comparator"]) != "decision1" for pair in pairs)
    ):
        raise ValueError("Pre-key pairs do not cover the Decision 2.0 roster")
    comparisons = plan.get("paired_ci_commands_after_prekey_freeze")
    if not isinstance(comparisons, list) or len(comparisons) != len(pairs):
        raise ValueError("Pre-key plan lacks one paired-CI command per candidate")
    by_candidate = {pair["candidate"]: pair for pair in pairs}
    seen_comparisons: set[str] = set()
    for entry in comparisons:
        if (
            not isinstance(entry, dict)
            or entry.get("candidate") not in by_candidate
            or entry["candidate"] in seen_comparisons
            or any(
                entry.get(name) != by_candidate[entry["candidate"]][name]
                for name in ("comparator", "size_relation", "rationale")
            )
            or not isinstance(entry.get("command"), str)
        ):
            raise ValueError("Paired-CI command does not match the frozen pair")
        seen_comparisons.add(entry["candidate"])
        tokens = shlex.split(entry["command"])
        for flag, expected in (("--replicates", "5000"), ("--seed", "20260927")):
            if tokens.count(flag) != 1 or tokens[
                tokens.index(flag) + 1 : tokens.index(flag) + 2
            ] != [expected]:
                raise ValueError("Paired-CI command changed its bootstrap policy")
    for row in roster:
        model = freeze["models"][row["key"]]
        if any(
            model[field] != row[planned]
            for field, planned in (
                ("model_id", "model_id"),
                ("revision", "revision"),
                ("native_model_sha256", "native_model_sha256"),
                ("adapter_sha256", "adapter_sha256"),
                ("calibration_sha256", "calibration_sha256"),
            )
        ):
            raise ValueError(f"{row['key']}: pre-key model identity differs from plan")
    audited_models = audit.get("models")
    if not isinstance(audited_models, dict) or set(audited_models) != keys:
        raise ValueError("Gold-free audit lacks the full model roster")
    for key in keys:
        if audited_models[key] != freeze["models"][key]["predictions_sha256"]:
            raise ValueError(
                f"{key}: audited full-panel predictions differ from freeze"
            )
    prompts = audit.get("prompt_sha256")
    if not isinstance(prompts, dict) or set(prompts) != set(PREDICTION_KEYS):
        raise ValueError("Gold-free audit lacks all three prompt panels")
    for name in PREDICTION_KEYS:
        _digest(prompts[name], f"{name} prompt SHA-256")
    if (
        freeze.get("prompt_sha256") != prompts
        or prompts["css"] != plan.get("css_prompts", {}).get("sha256")
        or prompts["public"] != plan.get("public_panel", {}).get("prompts_sha256")
        or freeze.get("raw_prediction_hashes_sha256") != audit.get("raw_hashes_sha256")
    ):
        raise ValueError("Gold-free prompt or prediction inventory differs")
    _digest(audit.get("raw_hashes_sha256"), "raw prediction hash list")
    parsed = _utc(freeze.get("prekey_frozen_at_utc"), "prekey_frozen_at_utc")
    _chronology(freeze, path, parsed)
    return plan


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
    freeze["_validated_plan"] = _prekey_evidence(freeze, path, keys)
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
    planned = {row["key"]: row for row in freeze["_validated_plan"]["model_roster"]}
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
        expected = planned[key]
        if entry["group"] != expected["group"] or entry["size_b"] != expected["size_b"]:
            raise ValueError(f"{key}: roster group or measured size differs from plan")
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
        typed, css = (_load(paths[name]) for name in SCORED_KEYS)
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
