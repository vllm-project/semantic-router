"""Fail-closed native model package for the two-axis JevArena v3 first release.

This keeps the v2 six-axis path immutable. It checks previously produced,
gold-free scorer/native receipts, source rights, exact model bytes, a separate
public231 ranking, and an external pre-key audit. It does not run inference or
prove that a declared independent reviewer actually reviewed the evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from jev_arena.arena_v3 import SCORER_SOURCE_PATHS
from jev_arena.arena_v3 import _css as score_css_axis
from jev_arena.arena_v3 import _freeze as checked_freeze
from jev_arena.arena_v3 import _typed as score_typed_axis
from jev_arena.compare_v3 import DEFAULT_REPLICATES, DEFAULT_SEED
from jev_arena.jevbench_public import SCORE_VERSION as PUBLIC_SCORE_VERSION
from scripts.eikos_stable_runtime_v3 import stable_backend

from . import adapter_runtime
from . import bundle_arena as common
from .generate_arena_v3 import ARTIFACTS, matched_models
from .generate_arena_v3 import VERSION as ARTIFACT_VERSION

VERSION = "decision2-jevarena-v3-release-bundle/1"
GATE_VERSION = "decision2-jevarena-v3-release-gate/1"
CHECK_VERSION = "decision2-jevarena-v3-release-check/1"
FREEZE_AUDIT_VERSION = "decision2-jevarena-v3-freeze-audit/2"
TIMESTAMP_LOG_VERSION = "decision2-jevarena-v3-timestamp-log/1"
SCORE_FAMILIES = ("typed", "css", "public")
GATE_CHECKS = (
    "candidate_freeze",
    "train_eval_overlap",
    "same_panel_evaluation",
    "rights_and_provenance",
    "native_parity",
    "release_thresholds",
)
POLICY = (
    Path(__file__).resolve().parents[1]
    / "research/jev-arena-v3-first-release-gates-2026-09-27.md"
)
EIKOS_COLLECTOR = (
    Path(__file__).resolve().parents[1] / "training/eikos/published_infer.py"
)
SCORERS = SCORER_SOURCE_PATHS


def _native_adapter_sha(native: dict[str, Any], family: str) -> str:
    if "collector_source_sha256" not in native:
        return common._sha(native.get("adapter_sha256"), f"{family} adapter")
    # Eikos' collector source is its frozen adapter identity. The full-run
    # receipt must retain the qualified backend and every original item.
    if (
        native.get("adapter_sha256") is not None
        or native.get("collector_source_sha256") != common.sha_file(EIKOS_COLLECTOR)
        or native.get("input_items")
        != {"typed": 1600, "css": 6547, "public": 231}[family]
        or native.get("evaluated_items") != native.get("input_items")
        or native.get("counts", {}).get("items") != native.get("input_items")
        or native.get("max_items") is not None
        or not stable_backend(native.get("runtime"))
    ):
        raise ValueError(f"{family}: Eikos native collector/runtime differs")
    return common._sha(native["collector_source_sha256"], f"{family} collector")


def _context_digest(
    record_sha: str,
    parity_sha: str,
    artifact_sha: str,
    arena_sha: str,
    public_sha: str,
    freeze_sha: str,
    score_binding: dict[str, dict[str, str]],
    comparison_binding: dict[str, str],
) -> str:
    payload = {
        "record": record_sha,
        "parity": parity_sha,
        "artifacts": artifact_sha,
        "arena": arena_sha,
        "public": public_sha,
        "freeze": freeze_sha,
        "scores": score_binding,
        "comparison": comparison_binding,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifacts(
    artifacts: Path,
    arena_rank: Path,
    public_rank: Path,
    model_id: str,
    revision: str,
    key: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        manifest.get("publication_version") != ARTIFACT_VERSION
        or manifest.get("phase") != "release"
        or manifest.get("arena_schema_version") != "jevarena-ranking/3"
    ):
        raise ValueError("First release needs v3 card artifacts")
    hashes = manifest.get("artifacts_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACTS):
        raise ValueError("V3 card artifact inventory is incomplete")
    for name in ARTIFACTS:
        path = artifacts / name
        if (
            path.is_symlink()
            or not path.is_file()
            or common.sha_file(path) != hashes[name]
        ):
            raise ValueError(f"V3 card artifact differs from generator: {name}")
        common._public_text(path.read_text(encoding="utf-8"), name)
    if manifest.get("ranking_sha256") != {
        "arena": common.sha_file(arena_rank),
        "jevbench_public": common.sha_file(public_rank),
    }:
        raise ValueError("V3 card ranks differ from generator inputs")
    arena, public = common._object(arena_rank), common._object(public_rank)
    models = matched_models(arena, public)
    if manifest.get("panel_sha256") != {
        "sealed_core": arena["panel_sha256"],
        "jevbench_public": public["panel_sha256"],
    }:
        raise ValueError("V3 card panel digests differ from ranked panels")
    if manifest.get("coverage") != {
        "sealed_core_items": 8147,
        "jevbench_public_items": 231,
        "public_items_in_arena_score": 0,
        "authored_items_in_arena_score": 0,
    }:
        raise ValueError("V3 card coverage is incomplete or mixes public items")
    if key not in models:
        raise ValueError("Selected model absent from matched v3/public rankings")
    row, peer = models[key]
    if (
        row["group"] != "decision2"
        or row["model_id"] != model_id
        or row["revision"] != revision
        or peer["group"] != "decision2"
    ):
        raise ValueError("Selected v3 card model identity differs from package")
    emitted = manifest.get("models")
    if not isinstance(emitted, list) or len(emitted) != len(models):
        raise ValueError("V3 card model roster is incomplete")
    if {item.get("key") for item in emitted if isinstance(item, dict)} != set(models):
        raise ValueError("V3 card model roster differs from rankings")
    for item in emitted:
        if not isinstance(item, dict):
            raise ValueError("V3 card model roster is malformed")
        ranked, public_row = models[item["key"]]
        if any(
            item.get(field) != value
            for field, value in (
                ("model_id", ranked["model_id"]),
                ("revision", ranked["revision"]),
                ("size_b", ranked["size_b"]),
                ("arena_rank", ranked["rank"]),
                ("jevbench_public_rank", public_row["rank"]),
            )
        ):
            raise ValueError("V3 card ranks or identities differ from scored reports")
    return arena, row, peer


def _score_inputs(
    paths: dict[str, dict[str, Path]],
    row: dict[str, Any],
    public: dict[str, Any],
    arena: dict[str, Any],
    public_rank: dict[str, Any],
    native_sha: str,
    model_id: str,
    revision: str,
    calibration_sha: str,
    adapter_version: str,
    package_manifest_sha: str | None,
    checkpoint_files: dict[str, str] | None,
) -> dict[str, dict[str, str]]:
    if not isinstance(paths, dict) or set(paths) != set(SCORE_FAMILIES):
        raise ValueError("V3 release requires typed, CSS and public231 native runs")
    result = {}
    for family in SCORE_FAMILIES:
        value = paths[family]
        if not isinstance(value, dict) or set(value) != {
            "score",
            "predictions",
            "native_manifest",
        }:
            raise ValueError(f"{family}: incomplete scored native run")
        score_path, predictions, native_path = (
            value[name] for name in ("score", "predictions", "native_manifest")
        )
        if any(
            path.is_symlink() or not path.is_file()
            for path in (score_path, predictions, native_path)
        ):
            raise ValueError(f"{family}: scored native run is not a regular file")
        score, native = common._object(score_path), common._object(native_path)
        score_sha = common.sha_file(score_path)
        prediction_sha = common.sha_file(predictions)
        native_receipt_sha = common.sha_file(native_path)
        wanted = (
            public["report_sha256"]
            if family == "public"
            else row["report_sha256"][family]
        )
        if score_sha != wanted:
            raise ValueError(f"{family}: scorer report differs from ranked result")
        if (
            score.get("predictions_sha256") != prediction_sha
            or native.get("predictions_sha256") != prediction_sha
        ):
            raise ValueError(f"{family}: predictions differ from scored native run")
        if family == "typed":
            if (
                score.get("schema_version") != "typed-decision-report/2"
                or score.get("split") != "final"
                or score.get("items") != 1600
                or score.get("model", {}).get("id") != model_id
                or score.get("model", {}).get("revision") != revision
                or score.get("gold_sha256")
                != arena["panel_sha256"]["typed_gold_sha256"]
            ):
                raise ValueError("Typed score is not the v3 FINAL panel")
            axis, tasks = score_typed_axis(score, model_id, revision)
            if (
                not math.isclose(axis, row["axes"]["typed"], abs_tol=1e-10)
                or tasks != row["task_scores"]["typed"]
            ):
                raise ValueError("Typed score differs from v3 ranking components")
        elif family == "css":
            if (
                score.get("score_schema_version") != "css-transfer-score/2"
                or score.get("gold_sha256") != arena["panel_sha256"]["css_gold_sha256"]
                or score.get("roles", {}).get("evaluation", {}).get("items") != 6547
            ):
                raise ValueError("CSS score is not the 15-task v3 FINAL panel")
            axis, tasks = score_css_axis(score)
            if (
                not math.isclose(axis, row["axes"]["transfer"], abs_tol=1e-10)
                or tasks != row["task_scores"]["transfer"]
            ):
                raise ValueError("CSS score differs from v3 ranking components")
        elif (
            score.get("score_version") != PUBLIC_SCORE_VERSION
            or score.get("items") != 231
            or score.get("model_id") != model_id
            or score.get("model_revision") != revision
            or score.get("prompts_sha256")
            != public_rank["panel_sha256"]["prompts_sha256"]
            or score.get("targets_sha256")
            != public_rank["panel_sha256"]["targets_sha256"]
            or score.get("panel_manifest_sha256")
            != public_rank["panel_sha256"]["panel_manifest_sha256"]
            or score.get("prediction_manifest_sha256") != native_receipt_sha
            or score.get("accuracy_all") != public["accuracy_all"]
            or score.get("valid") != public["valid"]
            or score.get("tier_macro_accuracy") != public["tier_macro_accuracy"]
            or any(
                score.get("tiers", {}).get(tier, {}).get("accuracy_all")
                != public["tiers"][tier]
                for tier in ("easy", "standard", "hard")
            )
        ):
            raise ValueError("JevBench score is not the pinned public231 panel")
        if family in {"typed", "css"} and score.get(
            "prediction_manifest_sha256"
        ) not in (None, native_receipt_sha):
            raise ValueError(f"{family}: scorer used another native receipt")
        native_cal = native.get("calibration_sha256") or native.get(
            "calibration", {}
        ).get("file_sha256")
        if (
            native.get("model_id") != model_id
            or native.get("model_revision") != revision
            or native.get("model_sha256") != native_sha
            or native.get("adapter_version") != adapter_version
            or native_cal != calibration_sha
        ):
            raise ValueError(f"{family}: native inference used another model package")
        adapter_sha = _native_adapter_sha(native, family)
        if adapter_sha != row["adapter_sha256"]:
            raise ValueError(f"{family}: native adapter differs from v3 freeze")
        if (
            package_manifest_sha is not None
            and native.get("package_manifest_sha256") != package_manifest_sha
        ):
            raise ValueError(
                f"{family}: native inference used another external package"
            )
        if (
            checkpoint_files is not None
            and native.get("model_files_sha256") != checkpoint_files
        ):
            raise ValueError(f"{family}: native inference used another external base")
        result[family] = {
            "score_sha256": score_sha,
            "predictions_sha256": prediction_sha,
            "native_manifest_sha256": native_receipt_sha,
            "adapter_sha256": adapter_sha,
        }
    return result


def _utc(value: Any, label: str) -> datetime:
    return common._utc(value, label)


def _gate(
    gate: dict[str, Any],
    *,
    record_sha: str,
    parity_sha: str,
    artifact_sha: str,
    arena_sha: str,
    public_sha: str,
    freeze_sha: str,
    native_sha: str,
    model_files_sha: str,
    model_id: str,
    revision: str,
) -> None:
    required = {
        "package_record_sha256": record_sha,
        "parity_receipt_sha256": parity_sha,
        "artifact_manifest_sha256": artifact_sha,
        "arena_rank_sha256": arena_sha,
        "jevbench_public_rank_sha256": public_sha,
        "pretest_freeze_sha256": freeze_sha,
        "native_model_sha256": native_sha,
        "model_files_sha256": model_files_sha,
    }
    if (
        gate.get("schema_version") != GATE_VERSION
        or gate.get("status") != "passed"
        or gate.get("model_id") != model_id
        or gate.get("model_revision") != revision
        or any(gate.get(key) != value for key, value in required.items())
    ):
        raise ValueError("V3 release gate is missing, blocked, or bound to other bytes")
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(GATE_CHECKS):
        raise ValueError("V3 release gate omits a required review")
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("status") != "passed":
            raise ValueError(f"V3 release review is blocked: {name}")
        common._sha(check.get("evidence_sha256"), f"{name} evidence")


def _external_evidence(
    record: dict[str, Any],
    gate: dict[str, Any],
    provenance_inputs: dict[str, Path],
    freeze_manifest: Path,
    gate_evidence: dict[str, Path],
    *,
    freeze: dict[str, Any],
    arena: dict[str, Any],
    row: dict[str, Any],
    candidate: dict[str, Any],
    score_binding: dict[str, dict[str, str]],
    comparison_binding: dict[str, str],
    context_sha: str,
) -> dict[str, str]:
    _, _, freeze_sha = common._json_snapshot(freeze_manifest)
    if (
        freeze_sha != arena["freeze_sha256"]
        or freeze.get("schema_version") != "jevarena-v3-freeze/2"
        or freeze.get("status") != "prekey_frozen"
        or freeze.get("panels") != arena["panel_sha256"]
        or freeze.get("models", {}).get(row["key"])
        != {
            "model_id": candidate["model_id"],
            "revision": candidate["model_revision"],
            "native_model_sha256": candidate["native_model_sha256"],
            "adapter_sha256": candidate["adapter_sha256"],
            "calibration_sha256": candidate["calibration_sha256"],
            "predictions_sha256": {
                name: score_binding[name]["predictions_sha256"]
                for name in SCORE_FAMILIES
            },
        }
    ):
        raise ValueError(
            "V3 pre-key freeze does not bind the selected package and predictions"
        )
    if not isinstance(freeze.get("candidate_lock_sha256"), str):
        raise ValueError("V3 pre-key freeze lacks a candidate lock")
    common._sha(freeze["candidate_lock_sha256"], "candidate lock")
    protocol_sha = common._sha(freeze.get("protocol_sha256"), "v3 protocol")
    if protocol_sha != common.sha_file(POLICY):
        raise ValueError(
            "Pre-key freeze does not bind the preregistered v3 release policy"
        )
    source_shas = freeze.get("score_sources_sha256")
    if not isinstance(source_shas, dict) or set(source_shas) != set(SCORERS):
        raise ValueError("V3 freeze lacks scoring source versions")
    for name, value in source_shas.items():
        if common._sha(value, f"{name} scorer") != common.sha_file(SCORERS[name]):
            raise ValueError(f"{name}: frozen scoring source changed")
    training = record["training"]
    required_provenance = {
        "data_manifest": training["data_manifest_sha256"],
        "run_provenance": training["run_provenance_sha256"],
        "training_code": training["training_code_sha256"],
    }
    if not isinstance(provenance_inputs, dict) or set(provenance_inputs) != set(
        required_provenance
    ):
        raise ValueError("V3 release needs exact private training provenance")
    if not isinstance(gate_evidence, dict) or set(gate_evidence) != set(GATE_CHECKS):
        raise ValueError("V3 release needs all six independent check receipts")
    observed = {}
    for name, path in {
        **provenance_inputs,
        **{f"gate:{key}": value for key, value in gate_evidence.items()},
    }.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V3 release evidence is missing or linked: {name}")
        actual = common.sha_file(path)
        wanted = (
            required_provenance[name]
            if name in required_provenance
            else gate["checks"][name.removeprefix("gate:")]["evidence_sha256"]
        )
        if actual != wanted:
            raise ValueError(f"V3 release evidence changed after review: {name}")
        observed[name] = actual
        if not name.startswith("gate:"):
            continue
        check_name = name.removeprefix("gate:")
        evidence = common._object(path)
        if (
            evidence.get("schema_version")
            != (
                FREEZE_AUDIT_VERSION
                if check_name == "candidate_freeze"
                else CHECK_VERSION
            )
            or evidence.get("status") != "passed"
            or evidence.get("check") != check_name
            or evidence.get("model_id") != candidate["model_id"]
            or evidence.get("model_revision") != candidate["model_revision"]
            or evidence.get("release_context_sha256") != context_sha
        ):
            raise ValueError(f"{check_name}: review is not bound to this v3 release")
        common._sha(evidence.get("reviewer_identity_sha256"), f"{check_name} reviewer")
        common._sha(
            evidence.get("source_evidence_sha256"), f"{check_name} source evidence"
        )
        reviewed = _utc(evidence.get("reviewed_at_utc"), f"{check_name} review time")
        if check_name == "candidate_freeze":
            if (
                evidence.get("pretest_freeze_sha256") != freeze_sha
                or evidence.get("candidate_lock_sha256")
                != freeze["candidate_lock_sha256"]
                or evidence.get("candidate") != candidate
                or evidence.get("protocol_sha256") != protocol_sha
            ):
                raise ValueError("Candidate audit differs from pre-key freeze")
            log_name = evidence.get("timestamp_log_path")
            if not isinstance(log_name, str) or not log_name:
                raise ValueError("External timestamp log path is missing")
            log_path = Path(log_name)
            if log_path.is_symlink() or not log_path.is_file():
                raise ValueError("External timestamp log is missing or linked")
            if common.sha_file(log_path) != common._sha(
                evidence.get("timestamp_log_sha256"), "external timestamp log"
            ):
                raise ValueError("External timestamp log digest differs")
            log = common._object(log_path)
            lock_time = _utc(
                evidence.get("candidate_locked_at_utc"), "candidate lock time"
            )
            prekey_time = _utc(
                freeze.get("prekey_frozen_at_utc"), "pre-key receipt time"
            )
            label_open = _utc(
                evidence.get("first_label_opened_at_utc"), "first label access"
            )
            seals = evidence.get("prediction_seals")
            if not isinstance(seals, dict) or set(seals) != set(freeze["models"]):
                raise ValueError("Pre-key prediction seals lack the full model roster")
            for model_key, panels in seals.items():
                if not isinstance(panels, dict) or set(panels) != set(SCORE_FAMILIES):
                    raise ValueError(
                        f"{model_key}: three prediction seals are required"
                    )
                for family, seal in panels.items():
                    expected = freeze["models"][model_key]["predictions_sha256"][family]
                    if (
                        not isinstance(seal, dict)
                        or seal.get("predictions_sha256") != expected
                        or (
                            model_key == row["key"]
                            and seal.get("native_manifest_sha256")
                            != score_binding[family]["native_manifest_sha256"]
                        )
                    ):
                        raise ValueError(
                            f"{model_key}.{family}: pre-key prediction seal differs"
                        )
                    sealed = _utc(
                        seal.get("sealed_at_utc"),
                        f"{model_key}.{family} prediction seal time",
                    )
                    if not lock_time < sealed < prekey_time < label_open < reviewed:
                        raise ValueError(
                            "Candidate lock, prediction, pre-key and label chronology is invalid"
                        )
            if log != {
                "schema_version": TIMESTAMP_LOG_VERSION,
                "candidate_lock_sha256": freeze["candidate_lock_sha256"],
                "candidate_locked_at_utc": evidence["candidate_locked_at_utc"],
                "prediction_seals": seals,
                "prekey_freeze_sha256": freeze_sha,
                "prekey_frozen_at_utc": freeze["prekey_frozen_at_utc"],
                "first_label_opened_at_utc": evidence["first_label_opened_at_utc"],
            }:
                raise ValueError("External timestamp log differs from reviewed events")
        elif check_name == "release_thresholds":
            if (
                evidence.get("predeclared_policy_sha256") != protocol_sha
                or evidence.get("comparison_inputs_sha256") != comparison_binding
            ):
                raise ValueError(
                    "Release threshold review is not bound to the frozen policy"
                )
    return observed


def _fraction(value: Any, label: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label}: expected a finite fraction")
    return float(value)


def _numeric_release_gate(
    *,
    pair_report: Path,
    old_typed_report: Path,
    old_css_report: Path,
    new_typed_report: Path,
    new_css_report: Path,
    artifact_manifest: dict[str, Any],
    arena: dict[str, Any],
    freeze: dict[str, Any],
    score_key: str,
    row: dict[str, Any],
    score_binding: dict[str, dict[str, str]],
    postkey_aggregate_priority: bool = False,
) -> dict[str, str]:
    """Recompute the strict or explicitly post-key numeric release rules."""
    paths = (pair_report, old_typed_report, old_css_report)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Joint paired CI and comparator score reports are required")
    pairs = [
        item
        for item in artifact_manifest.get("comparison_pairs", [])
        if item.get("new") == score_key
    ]
    if len(pairs) != 1:
        raise ValueError("V3 release needs one exact paired Decision 1.0 comparator")
    pair_spec = pairs[0]
    old_key = pair_spec["old"]
    old_rows = [item for item in arena["models"] if item.get("key") == old_key]
    if len(old_rows) != 1 or old_rows[0].get("group") != "decision1":
        raise ValueError("Frozen comparator is absent from the V3 roster")
    old_row = old_rows[0]
    frozen_pairs = [
        pair for pair in freeze["comparison_pairs"] if pair["candidate"] == score_key
    ]
    if len(frozen_pairs) != 1 or frozen_pairs[0]["comparator"] != old_key:
        raise ValueError("Publication comparator differs from pre-key pair")
    frozen_pair = frozen_pairs[0]
    planned = {item["key"]: item for item in freeze["_validated_plan"]["model_roster"]}
    if (
        row["size_b"] != planned[score_key]["size_b"]
        or old_row["size_b"] != planned[old_key]["size_b"]
    ):
        raise ValueError("Paired measured model sizes differ from pre-key plan")
    ratio = max(row["size_b"], old_row["size_b"]) / min(
        row["size_b"], old_row["size_b"]
    )
    if frozen_pair["size_relation"] == "same":
        if ratio > 1.25 + 1e-12:
            raise ValueError("Same-size comparator exceeds measured 1.25 ratio")
    elif frozen_pair["size_relation"] == "nearest":
        if (
            not isinstance(frozen_pair["rationale"], str)
            or len(frozen_pair["rationale"].strip()) < 20
        ):
            raise ValueError("Nearest-size comparator lacks a predeclared rationale")
    else:
        raise ValueError("Unknown frozen comparator size relation")
    report_hashes = pair_spec.get("report_sha256", {})
    if (
        common.sha_file(pair_report) != report_hashes.get("joint_comparison")
        or common.sha_file(old_typed_report) != report_hashes.get("old_typed_report")
        or common.sha_file(old_css_report) != report_hashes.get("old_transfer_report")
        or common.sha_file(new_typed_report) != report_hashes.get("new_typed_report")
        or common.sha_file(new_css_report) != report_hashes.get("new_transfer_report")
    ):
        raise ValueError("Comparator score reports differ from paired card evidence")
    new_typed, old_typed = (
        common._object(new_typed_report),
        common._object(old_typed_report),
    )
    new_css, old_css = common._object(new_css_report), common._object(old_css_report)
    aggregate = common._object(pair_report)
    if (
        aggregate.get("schema_version") != "jevarena-v3-paired-aggregate/1"
        or aggregate.get("models")
        != {"left": row["model_id"], "right": old_row["model_id"]}
        or aggregate.get("typed_gold_sha256")
        != arena["panel_sha256"]["typed_gold_sha256"]
        or aggregate.get("css_gold_sha256") != arena["panel_sha256"]["css_gold_sha256"]
        or aggregate.get("predictions_sha256")
        != {
            "left": {
                "typed": score_binding["typed"]["predictions_sha256"],
                "css": score_binding["css"]["predictions_sha256"],
            },
            "right": {
                "typed": old_typed.get("predictions_sha256"),
                "css": old_css.get("predictions_sha256"),
            },
        }
        or type(aggregate.get("replicates")) is not int
        or aggregate["replicates"] != DEFAULT_REPLICATES
        or type(aggregate.get("seed")) is not int
        or aggregate["seed"] != DEFAULT_SEED
        or aggregate.get("coverage")
        != {
            "typed_items": 1600,
            "typed_independent_groups": 400,
            "css_evaluation_items": 6547,
            "css_evaluation_tasks": 15,
        }
    ):
        raise ValueError("Joint v3 paired bootstrap is missing or unbound")
    if aggregate.get("source_sha256") != {
        name: common.sha_file(path) for name, path in SCORERS.items()
    }:
        raise ValueError("Joint paired bootstrap scoring source differs from freeze")
    bootstrap = aggregate.get("bootstrap", {})
    if (
        bootstrap.get("fixed_css_label_universe") is not True
        or bootstrap.get("confidence_level") != 0.95
    ):
        raise ValueError("Joint paired bootstrap used another sampling policy")
    panel_bytes = json.dumps(
        arena["panel_sha256"], sort_keys=True, separators=(",", ":")
    ).encode()
    if aggregate.get("panel_sha256") != hashlib.sha256(panel_bytes).hexdigest():
        raise ValueError("Joint paired bootstrap panel digest differs")
    point = aggregate.get("point", {})
    new_point, old_point, delta = (
        point.get(name, {}) for name in ("left", "right", "delta")
    )
    expected_new = {
        "T": row["axes"]["typed"],
        "H": row["axes"]["transfer"],
        "score": row["score"],
    }
    expected_old = {
        "T": old_row["axes"]["typed"],
        "H": old_row["axes"]["transfer"],
        "score": old_row["score"],
    }
    for name, value in expected_new.items():
        if type(new_point.get(name)) not in (int, float) or not math.isclose(
            new_point[name], value, abs_tol=1e-9
        ):
            raise ValueError("Joint paired candidate point differs from v3 ranking")
    for name, value in expected_old.items():
        if type(old_point.get(name)) not in (int, float) or not math.isclose(
            old_point[name], value, abs_tol=1e-9
        ):
            raise ValueError("Joint paired comparator point differs from v3 ranking")
        if type(delta.get(name)) not in (int, float) or not math.isclose(
            delta[name], expected_new[name] - value, abs_tol=1e-9
        ):
            raise ValueError("Joint paired delta differs from same-panel scores")
    if postkey_aggregate_priority:
        if expected_new["score"] - expected_old["score"] < 3.0 - 1e-12:
            raise ValueError("Post-key aggregate gain is below +3.0 points")
    elif not (
        expected_new["T"] > expected_old["T"]
        and expected_new["H"] > expected_old["H"]
        and expected_new["score"] > expected_old["score"]
    ):
        raise ValueError("V3 first-release T/H or aggregate point improvement failed")
    ci = aggregate.get("ci95", {})
    if (
        type(ci.get("low")) not in (int, float)
        or type(ci.get("high")) not in (int, float)
        or not math.isfinite(ci["low"])
        or not math.isfinite(ci["high"])
        or not 0 < ci["low"] <= ci["high"]
    ):
        raise ValueError("V3 aggregate paired 95% lower bound must be positive")
    new_types, old_types = new_typed.get("by_type"), old_typed.get("by_type")
    if (
        not isinstance(new_types, dict)
        or not isinstance(old_types, dict)
        or set(new_types) != {"choice", "noul", "score"}
        or set(old_types) != set(new_types)
    ):
        raise ValueError("Typed comparator lacks all three native task slices")
    for kind in ("choice", "noul", "score"):
        candidate = _fraction(new_types[kind].get("accuracy_all"), f"{kind} candidate")
        baseline = _fraction(old_types[kind].get("accuracy_all"), f"{kind} comparator")
        if not postkey_aggregate_priority and candidate < baseline - 0.02 - 1e-12:
            raise ValueError(f"V3 first-release {kind} slice regression exceeded 0.02")
    new_overall, old_overall = (
        new_typed.get("overall", {}),
        old_typed.get("overall", {}),
    )
    if new_overall.get("n") != 2000 or old_overall.get("n") != 2000:
        raise ValueError("Typed comparator lacks 2,000 answers on 1,600 items")
    for label, new_invalid, old_invalid in (
        (
            "typed",
            new_overall.get("invalid_or_missing_n"),
            old_overall.get("invalid_or_missing_n"),
        ),
        (
            "css",
            6547
            - new_css.get("roles", {}).get("evaluation", {}).get("valid_items", -1),
            6547
            - old_css.get("roles", {}).get("evaluation", {}).get("valid_items", -1),
        ),
    ):
        denominator = 2000 if label == "typed" else 6547
        if any(
            type(value) is not int or not 0 <= value <= denominator
            for value in (new_invalid, old_invalid)
        ):
            raise ValueError(f"{label}: invalid/missing counts are malformed")
        if (
            new_invalid / denominator
            > max(0.02, old_invalid / denominator + 0.01) + 1e-12
        ):
            raise ValueError(f"{label}: invalid/missing guardrail failed")
    new_brier = _fraction(new_overall.get("brier"), "candidate typed Brier")
    old_brier = _fraction(old_overall.get("brier"), "comparator typed Brier")
    probability_counts = (
        new_overall.get("probability_n"),
        old_overall.get("probability_n"),
    )
    if any(
        type(value) is not int
        or not 0 <= value <= 2000 - report["invalid_or_missing_n"]
        for value, report in zip(
            probability_counts, (new_overall, old_overall), strict=True
        )
    ):
        raise ValueError("Typed Brier needs complete probability coverage counts")
    new_adjusted = (
        probability_counts[0] * new_brier + 2000 - probability_counts[0]
    ) / 2000
    old_adjusted = (
        probability_counts[1] * old_brier + 2000 - probability_counts[1]
    ) / 2000
    if new_adjusted > old_adjusted + 0.03 + 1e-12:
        raise ValueError(
            "V3 first-release coverage-adjusted typed Brier guardrail failed"
        )
    return {
        "aggregate_pair_report_sha256": common.sha_file(pair_report),
        "old_typed_report_sha256": common.sha_file(old_typed_report),
        "old_css_report_sha256": common.sha_file(old_css_report),
    }


def _typed_probability_table(
    candidate: dict[str, Any], comparator: dict[str, Any]
) -> str:
    values = []
    for report in (candidate, comparator):
        overall = report["overall"]
        count = overall["probability_n"]
        brier = overall["brier"]
        adjusted = (count * brier + 2000 - count) / 2000
        values.append((count, brier, adjusted))
    return "\n".join(
        (
            "| Measure | Decision 2.0 | Paired Decision 1.0 |",
            "| --- | ---: | ---: |",
            f"| Accepted probability answers | {values[0][0]:,}/2,000 | {values[1][0]:,}/2,000 |",
            f"| Brier on accepted answers | {values[0][1]:.4f} | {values[1][1]:.4f} |",
            f"| Coverage-adjusted Brier¹ | {values[0][2]:.4f} | {values[1][2]:.4f} |",
            "",
            "¹ Assigns normalized Brier 1 to each answer without an accepted probability; lower is better. The 1,600 typed items contain 2,000 scored answers. Invalid answers still count as failures in the capability score.",
        )
    )


def _card(
    model_id: str,
    record: dict[str, Any],
    row: dict[str, Any],
    count: int,
    score_table: str,
    probability_table: str,
) -> str:
    rights = record["rights"]
    sources = "\n".join(
        "| "
        + " | ".join(
            common._md(source[field])
            for field in (
                "name",
                "license",
                "attribution",
                "use_scope",
                "redistribution",
            )
        )
        + " |"
        for source in rights["sources"]
    )
    languages = "\n".join(
        f"| {common._md(language)} | {rows:,} | {rows / record['training']['train_rows']:.1%} |"
        for language, rows in sorted(record["training"]["language_counts"].items())
    )
    limitations = "\n".join(f"- {common._md(item)}" for item in record["limitations"])
    overlap = (
        "\n".join(f"- {common._md(item)}" for item in record["known_overlap"])
        or "- None declared."
    )
    external_note = ""
    if record["architecture"] == common.EXTERNAL_ADAPTER:
        external_note = (
            "This repository keeps the scored adapter and Decision head; its pinned "
            "upstream base is an external dependency. Load with the bundled native "
            "Decision2 runtime. A generic AutoModel load omits the decision head.\n\n"
        )
    return f"""---
license: {record["license_id"]}
{("license_name: noncommercial-research-terms" + chr(10)) if record["license_id"] == "other" else ""}base_model: {record["base_model"]["id"]}
tags:
- decision-model
- typed-decision
- jevarena-v3
---

![Decision 2.0 crossroads fox mosaic sticker](decision-2-sticker-crossroads-fox-v5.png)

# {model_id}

Native architecture: `{record["architecture"]}`. Actual loaded parameters:
**{count:,}**. Source model: [{record["base_model"]["id"]}](https://huggingface.co/{record["base_model"]["id"]})
at immutable revision `{record["base_model"]["revision"]}`.

{external_note}## Same-panel first-release evaluation

JevArena v3 sealed-core rank **#{row["rank"]}**, score **{row["score"]:.2f}**.
The headline uses only typed FINAL 1,600 and 15-task human transfer 6,547,
for **8,147** sealed-core items. Invalid and missing answers count as failures.
The JevBench public 231-item rerun is required but separate, and is not an
official closed-set JevBench rank. Authored questions are reserved for v3.1;
none were included in this v3 score. Decision Bench and Decision Index are not
first-release score axes.

![JevArena v3 ranking](jevarena-rank.svg)
![JevArena v3 parameter Pareto plot](jevarena-pareto.svg)
![JevArena v3 component matrix](jevarena-axis-matrix.svg)
![JevArena v3 model by task matrix](jevarena-task-matrix.svg)
![JevBench public ranking](jevbench-public-rank.svg)
![JevBench public parameter Pareto plot](jevbench-public-pareto.svg)

{score_table.rstrip()}

## Typed probability quality on the same sealed panel

{probability_table}

The six figures, scorer versions, frozen panel digests and paired intervals are
bound in `card-artifacts/manifest.json`. Robustness, calibration, language,
invalidity and efficiency are disclosed separately where measured. Runtime
speed and cost are comparable only under matched conditions.

## Training, rights and limitations

TRAIN {record["training"]["train_rows"]:,}; SELECT {record["training"]["select_rows"]:,};
CAL {record["training"]["cal_rows"]:,}. Selection policy:
{common._md(record["training"]["selection_policy"])}. Scope: `{rights["scope"]}`.
No raw upstream text or benchmark labels are bundled.

| TRAIN language | Rows | Share |
| --- | ---: | ---: |
{languages}

Evaluation language coverage: {common._md(record["evaluation_language_scope"])}.

| Source | Terms | Attribution | Use | Redistribution |
| --- | --- | --- | --- | --- |
{sources}

Known training/evaluation overlap:
{overlap}

Limitations:
{limitations}

`PACKAGE_MANIFEST.json` binds this card, native model, provenance, package
parity and all three scored native runs. The packager checks reviewed release
receipts and exact bytes; it does not rerun GPU inference or independently
authenticate the declared reviewer and timestamp log.
"""


def assemble(
    *,
    model_dir: Path,
    artifacts: Path,
    arena_rank: Path,
    public_rank: Path,
    package_record: Path,
    parity_receipt: Path,
    release_gate: Path,
    provenance_inputs: dict[str, Path],
    freeze_manifest: Path,
    gate_evidence: dict[str, Path],
    score_inputs: dict[str, dict[str, Path]],
    score_key: str,
    output: Path,
    aggregate_pair_report: Path,
    old_typed_report: Path,
    old_css_report: Path,
    base_source: Path | None = None,
    adapter_source_parity_receipt: Path | None = None,
) -> dict[str, Any]:
    for path in (
        model_dir,
        artifacts,
        arena_rank,
        public_rank,
        package_record,
        parity_receipt,
        release_gate,
        freeze_manifest,
    ):
        if path.is_symlink():
            raise ValueError("V3 release input symlinks are not allowed")
    model_dir, artifacts, output = (
        model_dir.resolve(strict=True),
        artifacts.resolve(strict=True),
        output.resolve(),
    )
    if (
        output.exists()
        or output.is_relative_to(model_dir)
        or output.is_relative_to(artifacts)
    ):
        raise FileExistsError("V3 release output exists or is inside input")
    record, record_bytes, record_sha = common._json_snapshot(package_record)
    parity, parity_bytes, parity_sha = common._json_snapshot(parity_receipt)
    gate, gate_bytes, gate_sha = common._json_snapshot(release_gate)
    artifact_manifest, artifact_bytes, artifact_sha = common._json_snapshot(
        artifacts / "manifest.json"
    )
    arena_sha, public_sha, freeze_sha = (
        common.sha_file(path) for path in (arena_rank, public_rank, freeze_manifest)
    )
    model_id, revision = record.get("model_id"), record.get("model_revision")
    if (
        not isinstance(model_id, str)
        or not common.MODEL_ID.fullmatch(model_id)
        or not isinstance(revision, str)
        or not revision
    ):
        raise ValueError("V3 release needs a valid Decision 2.0 model ID and revision")
    common._rights(record, model_id, revision)
    architecture = record.get("architecture")
    external = architecture == common.EXTERNAL_ADAPTER
    if external and (base_source is None or adapter_source_parity_receipt is None):
        raise ValueError("External adapter release needs pinned base and source parity")
    if not external and (
        base_source is not None or adapter_source_parity_receipt is not None
    ):
        raise ValueError("External adapter inputs are invalid for this architecture")
    files = common._inventory(model_dir, allow_adapter_metadata=external)
    if record.get("model_files_sha256") != files:
        raise ValueError("Native model files differ from reviewed release record")
    common._profile(model_dir, architecture, files)
    external_manifest = None
    source_parity_sha = None
    checkpoint_files = None
    if external:
        assert base_source is not None and adapter_source_parity_receipt is not None
        external_manifest, count, checkpoint_files = common._external_adapter_contract(
            model_dir, base_source, record, files
        )
        source_parity_sha = common._adapter_source_parity(
            adapter_source_parity_receipt,
            external_manifest,
            files["MODEL_MANIFEST.json"],
        )
    else:
        count = common._parameter_count(
            model_dir,
            record.get("active_weight_files"),
            record.get("support_weight_files"),
            record.get("non_parameter_tensors"),
            files,
        )
    if (
        type(record.get("parameter_count")) is not int
        or record["parameter_count"] != count
        or not common._size_compatible(count, model_id)
    ):
        raise ValueError("Declared and actual model parameter counts differ")
    native_sha = common._native_identity(model_dir, record, files, external_manifest)
    calibration_name = record.get("calibration_file")
    allowed = (
        {"calib.json"}
        if architecture == "qwen3.5-semif"
        else (
            {"calib.json", "calibration.json"}
            if architecture == "encoder-decision"
            else {"calibration.json"}
        )
    )
    if calibration_name not in allowed or record.get("calibration_sha256") != files.get(
        calibration_name
    ):
        raise ValueError("Native CAL differs from reviewed package")
    arena, row, peer = _artifacts(
        artifacts,
        arena_rank,
        public_rank,
        model_id,
        revision,
        score_key,
        artifact_manifest,
    )
    freeze = checked_freeze(
        freeze_manifest,
        {"freeze_sha256": arena["freeze_sha256"]},
        {item["key"] for item in arena["models"]},
    )
    public_rank_obj = common._object(public_rank)
    if not math.isclose(row["size_b"], count / 1e9, rel_tol=0, abs_tol=1e-9):
        raise ValueError("V3 score parameter count differs from loaded model")
    if (
        row["native_model_sha256"] != native_sha
        or row["calibration_sha256"] != files[calibration_name]
    ):
        raise ValueError("V3 pre-key candidate differs from native model or CAL")
    files_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = _score_inputs(
        score_inputs,
        row,
        peer,
        arena,
        public_rank_obj,
        native_sha,
        model_id,
        revision,
        files[calibration_name],
        record["native_adapter_version"],
        files["MODEL_MANIFEST.json"] if external else None,
        checkpoint_files,
    )
    comparison_binding = _numeric_release_gate(
        pair_report=aggregate_pair_report,
        old_typed_report=old_typed_report,
        old_css_report=old_css_report,
        new_typed_report=score_inputs["typed"]["score"],
        new_css_report=score_inputs["css"]["score"],
        artifact_manifest=artifact_manifest,
        arena=arena,
        freeze=freeze,
        score_key=score_key,
        row=row,
        score_binding=binding,
    )
    common._parity(
        parity, model_id, revision, native_sha, files_digest, files[calibration_name]
    )
    _gate(
        gate,
        record_sha=record_sha,
        parity_sha=parity_sha,
        artifact_sha=artifact_sha,
        arena_sha=arena_sha,
        public_sha=public_sha,
        freeze_sha=freeze_sha,
        native_sha=native_sha,
        model_files_sha=files_digest,
        model_id=model_id,
        revision=revision,
    )
    candidate = {
        "model_id": model_id,
        "model_revision": revision,
        "native_model_sha256": native_sha,
        "model_files_sha256": files_digest,
        "calibration_sha256": files[calibration_name],
        "adapter_version": record["native_adapter_version"],
        "adapter_sha256": row["adapter_sha256"],
    }
    context_sha = _context_digest(
        record_sha,
        parity_sha,
        artifact_sha,
        arena_sha,
        public_sha,
        freeze_sha,
        binding,
        comparison_binding,
    )
    external_hashes = _external_evidence(
        record,
        gate,
        provenance_inputs,
        freeze_manifest,
        gate_evidence,
        freeze=freeze,
        arena=arena,
        row=row,
        candidate=candidate,
        score_binding=binding,
        comparison_binding=comparison_binding,
        context_sha=context_sha,
    )
    for name, payload in (
        ("release-record.json", record_bytes),
        ("native-parity.json", parity_bytes),
        ("release-gate.json", gate_bytes),
    ):
        common._public_text(payload.decode("utf-8"), name)
    sticker = Path(__file__).with_name("decision-2-sticker-crossroads-fox-v5.png")
    if sticker.is_symlink() or not sticker.is_file():
        raise ValueError("Decision 2.0 sticker asset is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, expected in files.items():
            target = stage / "native" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_dir / name, target)
            if common.sha_file(target) != expected:
                raise ValueError("Native model changed during V3 packaging")
        if architecture in {"qwen3.5-decision-head", "qwen3.8-decision-head"}:
            runtime = common._qwen_runtime_contract(stage / "native", native_sha)
            runtime["files_sha256"] = {
                name: common.sha_file(stage / "native" / name) for name in files
            }
            (stage / "native" / "MODEL_MANIFEST.json").write_text(
                json.dumps(
                    runtime,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            common._verify_qwen_runtime(stage / "native")
        elif external:
            assert base_source is not None
            from .adapter_bundle import _verify_staged_runtime

            _verify_staged_runtime(stage / "native", base_source)
        (stage / "card-artifacts").mkdir()
        for name in (*ARTIFACTS, "manifest.json"):
            target = stage / "card-artifacts" / name
            if name == "manifest.json":
                target.write_bytes(artifact_bytes)
                wanted = artifact_sha
            else:
                shutil.copyfile(artifacts / name, target)
                wanted = artifact_manifest["artifacts_sha256"][name]
            if common.sha_file(target) != wanted:
                raise ValueError(f"V3 artifact changed during packaging: {name}")
            if name in ARTIFACTS:
                shutil.copyfile(target, stage / name)
        for payload, name in (
            (record_bytes, "release-record.json"),
            (parity_bytes, "native-parity.json"),
            (gate_bytes, "release-gate.json"),
        ):
            (stage / name).write_bytes(payload)
        shutil.copyfile(sticker, stage / sticker.name)
        (stage / "README.md").write_text(
            _card(
                model_id,
                record,
                row,
                count,
                (stage / "score-table.md").read_text(encoding="utf-8"),
                _typed_probability_table(
                    common._object(score_inputs["typed"]["score"]),
                    common._object(old_typed_report),
                ),
            ),
            encoding="utf-8",
        )
        public_files = {
            path.relative_to(stage).as_posix(): common.sha_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "bundle_version": VERSION,
            "model_id": model_id,
            "model_revision": revision,
            "architecture": architecture,
            "parameter_count": count,
            "native_model_sha256": native_sha,
            "model_files_sha256": files_digest,
            "artifact_manifest_sha256": artifact_sha,
            "panel_sha256": artifact_manifest["panel_sha256"],
            "package_record_sha256": record_sha,
            "parity_receipt_sha256": parity_sha,
            "release_gate_sha256": gate_sha,
            "score_inputs_sha256": binding,
            "comparison_inputs_sha256": comparison_binding,
            "release_context_sha256": context_sha,
            "files_sha256": public_files,
        }
        if external:
            assert external_manifest is not None and source_parity_sha is not None
            manifest["external_base"] = external_manifest["base"]
            manifest["adapter_manifest_sha256"] = files["MODEL_MANIFEST.json"]
            manifest["adapter_source_parity_sha256"] = source_parity_sha
        (stage / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        verify(stage, base_source=base_source)
        for path, digest in (
            (package_record, record_sha),
            (parity_receipt, parity_sha),
            (release_gate, gate_sha),
            (artifacts / "manifest.json", artifact_sha),
            (arena_rank, arena_sha),
            (public_rank, public_sha),
            (freeze_manifest, freeze_sha),
            (aggregate_pair_report, comparison_binding["aggregate_pair_report_sha256"]),
            (old_typed_report, comparison_binding["old_typed_report_sha256"]),
            (old_css_report, comparison_binding["old_css_report_sha256"]),
        ):
            if (
                path.is_symlink()
                or not path.is_file()
                or common.sha_file(path) != digest
            ):
                raise ValueError("Reviewed V3 input changed during packaging")
        for name, path in {
            **provenance_inputs,
            **{f"gate:{key}": value for key, value in gate_evidence.items()},
        }.items():
            if (
                path.is_symlink()
                or not path.is_file()
                or common.sha_file(path) != external_hashes[name]
            ):
                raise ValueError(
                    "External V3 release evidence changed during packaging"
                )
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify(root: Path, *, base_source: Path | None = None) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("V3 release package must be a regular directory")
    manifest = common._object(root / "PACKAGE_MANIFEST.json")
    if manifest.get("bundle_version") != VERSION:
        raise ValueError("Unknown V3 release package version")
    expected = manifest.get("files_sha256")
    if (
        not isinstance(expected, dict)
        or not expected
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError("V3 package inventory is missing or linked")
    external = manifest.get("architecture") == common.EXTERNAL_ADAPTER
    if external:
        native = adapter_runtime._inventory(root / "native", ignore_bytecode=True)
        actual = {f"native/{name}": digest for name, digest in native.items()}
        actual.update(
            {
                path.relative_to(root).as_posix(): common.sha_file(path)
                for path in root.rglob("*")
                if path.is_file()
                and not path.is_relative_to(root / "native")
                and path != root / "PACKAGE_MANIFEST.json"
            }
        )
    else:
        if base_source is not None:
            raise ValueError("Unexpected external base supplied for full V3 package")
        actual = {
            path.relative_to(root).as_posix(): common.sha_file(path)
            for path in root.rglob("*")
            if path.is_file() and path != root / "PACKAGE_MANIFEST.json"
        }
    if actual != expected:
        raise ValueError("V3 package file inventory changed")
    model_files = {
        name.removeprefix("native/"): digest
        for name, digest in actual.items()
        if name.startswith("native/")
        and (external or name != "native/MODEL_MANIFEST.json")
    }
    files_sha = hashlib.sha256(
        json.dumps(model_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if files_sha != manifest.get("model_files_sha256"):
        raise ValueError("V3 native model inventory changed")
    record = common._object(root / "release-record.json")
    parity = common._object(root / "native-parity.json")
    gate = common._object(root / "release-gate.json")
    artifact = common._object(root / "card-artifacts/manifest.json")
    model_id, revision, count = (
        manifest.get("model_id"),
        manifest.get("model_revision"),
        manifest.get("parameter_count"),
    )
    if (
        not isinstance(model_id, str)
        or not common.MODEL_ID.fullmatch(model_id)
        or not isinstance(revision, str)
        or not revision
        or type(count) is not int
        or count <= 0
        or not common._size_compatible(count, model_id)
        or record.get("parameter_count") != count
        or record.get("model_files_sha256") != model_files
        or record.get("native_identity", {}).get("sha256")
        != manifest.get("native_model_sha256")
        or record.get("architecture") != manifest.get("architecture")
    ):
        raise ValueError("V3 publication manifest or record binding changed")
    common._rights(record, model_id, revision)
    calibration = record.get("calibration_file")
    calibration_sha = actual.get(f"native/{calibration}")
    if calibration_sha is None or calibration_sha != record.get("calibration_sha256"):
        raise ValueError("V3 package calibration differs from release record")
    if (
        artifact.get("publication_version") != ARTIFACT_VERSION
        or artifact.get("phase") != "release"
        or artifact.get("arena_schema_version") != "jevarena-ranking/3"
        or artifact.get("panel_sha256") != manifest.get("panel_sha256")
        or actual.get("card-artifacts/manifest.json")
        != manifest.get("artifact_manifest_sha256")
        or not isinstance(artifact.get("artifacts_sha256"), dict)
        or set(artifact["artifacts_sha256"]) != set(ARTIFACTS)
        or any(
            actual.get(f"card-artifacts/{name}") != artifact["artifacts_sha256"][name]
            or actual.get(name) != artifact["artifacts_sha256"][name]
            for name in ARTIFACTS
        )
    ):
        raise ValueError("V3 card artifacts or duplicated figures changed")
    if (
        actual.get("release-record.json") != manifest.get("package_record_sha256")
        or actual.get("native-parity.json") != manifest.get("parity_receipt_sha256")
        or actual.get("release-gate.json") != manifest.get("release_gate_sha256")
    ):
        raise ValueError("V3 release receipts differ from public manifest")
    common._parity(
        parity,
        model_id,
        revision,
        manifest["native_model_sha256"],
        files_sha,
        calibration_sha,
    )
    ranks = artifact.get("ranking_sha256", {})
    _gate(
        gate,
        record_sha=actual["release-record.json"],
        parity_sha=actual["native-parity.json"],
        artifact_sha=actual["card-artifacts/manifest.json"],
        arena_sha=ranks.get("arena"),
        public_sha=ranks.get("jevbench_public"),
        freeze_sha=gate.get("pretest_freeze_sha256"),
        native_sha=manifest["native_model_sha256"],
        model_files_sha=files_sha,
        model_id=model_id,
        revision=revision,
    )
    score_binding = manifest.get("score_inputs_sha256")
    if not isinstance(score_binding, dict) or set(score_binding) != set(SCORE_FAMILIES):
        raise ValueError("V3 package lacks all three scored native runs")
    for family, fields in score_binding.items():
        if not isinstance(fields, dict) or set(fields) != {
            "score_sha256",
            "predictions_sha256",
            "native_manifest_sha256",
            "adapter_sha256",
        }:
            raise ValueError(f"{family}: incomplete native score binding")
        for key, value in fields.items():
            common._sha(value, f"{family}.{key}")
    comparison_binding = manifest.get("comparison_inputs_sha256")
    if not isinstance(comparison_binding, dict) or set(comparison_binding) != {
        "aggregate_pair_report_sha256",
        "old_typed_report_sha256",
        "old_css_report_sha256",
    }:
        raise ValueError("V3 package lacks the joint paired comparison receipt")
    for name, value in comparison_binding.items():
        common._sha(value, name)
    if manifest.get("release_context_sha256") != _context_digest(
        actual["release-record.json"],
        actual["native-parity.json"],
        actual["card-artifacts/manifest.json"],
        ranks["arena"],
        ranks["jevbench_public"],
        gate["pretest_freeze_sha256"],
        score_binding,
        comparison_binding,
    ):
        raise ValueError("V3 publication release context changed")
    if manifest.get("architecture") in {
        "qwen3.5-decision-head",
        "qwen3.8-decision-head",
    }:
        common._verify_qwen_runtime(root / "native")
    elif external:
        if base_source is not None:
            from .adapter_bundle import _verify_staged_runtime

            _verify_staged_runtime(root / "native", base_source)
        inner = adapter_runtime.verify_bundle(root / "native")
        base = inner["base"]
        if (
            manifest.get("external_base") != base
            or manifest.get("adapter_manifest_sha256")
            != actual.get("native/MODEL_MANIFEST.json")
            or manifest.get("native_model_sha256") != inner["model_sha256"]
            or manifest.get("parameter_count") != inner["parameter_count"]
            or record.get("base_model", {}).get("id") != base["repo_id"]
            or record.get("base_model", {}).get("revision") != base["revision"]
        ):
            raise ValueError("V3 external base or adapter binding changed")
        common._sha(manifest.get("adapter_source_parity_sha256"), "source parity")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--base-source", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify(args.verify, base_source=args.base_source)
    else:
        if args.config is None or args.output is None:
            parser.error("--config and --output are required for assembly")
        config = common._object(args.config)
        base = args.config.parent

        def path(name: str) -> Path:
            value = Path(config[name])
            return value if value.is_absolute() else base / value

        def mapping(name: str) -> dict[str, Path]:
            if not isinstance(config.get(name), dict):
                raise ValueError(f"{name} must map receipt names to paths")
            return {
                key: (Path(value) if Path(value).is_absolute() else base / value)
                for key, value in config[name].items()
            }

        score_paths = {}
        for family, files in config["score_inputs"].items():
            score_paths[family] = {
                key: (Path(value) if Path(value).is_absolute() else base / value)
                for key, value in files.items()
            }
        result = assemble(
            model_dir=path("model_dir"),
            artifacts=path("artifacts"),
            arena_rank=path("arena_rank"),
            public_rank=path("public_rank"),
            package_record=path("package_record"),
            parity_receipt=path("parity_receipt"),
            release_gate=path("release_gate"),
            provenance_inputs=mapping("provenance_inputs"),
            freeze_manifest=path("freeze_manifest"),
            gate_evidence=mapping("gate_evidence"),
            score_inputs=score_paths,
            score_key=config["score_key"],
            output=args.output,
            aggregate_pair_report=path("aggregate_pair_report"),
            old_typed_report=path("old_typed_report"),
            old_css_report=path("old_css_report"),
            base_source=path("base_source") if "base_source" in config else None,
            adapter_source_parity_receipt=(
                path("adapter_source_parity_receipt")
                if "adapter_source_parity_receipt" in config
                else None
            ),
        )
    print(
        json.dumps({"model_id": result["model_id"], "version": VERSION}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
