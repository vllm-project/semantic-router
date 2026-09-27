"""Plan and audit a JevArena v3 first release without reading FINAL labels.

The planner validates completed Decision 2.0 selection/CAL/package lineage,
gold-free CSS prompts, a predeclared comparison roster, and source identities.
It only prints commands. The optional prediction audit reads gold-free prompts
and predictions, never either held-out answer file. A separate, later pre-key
freeze must bind the prediction and gold digests before any scoring command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any

from jev_arena.jevbench_public import (
    BUILD_VERSION as PUBLIC_BUILD_VERSION,
)
from jev_arena.jevbench_public import (
    FILES as PUBLIC_FILES,
)
from jev_arena.jevbench_public import (
    SOURCE_REVISION as PUBLIC_SOURCE_REVISION,
)
from jev_arena.jevbench_public import (
    SOURCE_URL as PUBLIC_SOURCE_URL,
)
from scripts.baseline_attestation_v3 import verify_attestation
from scripts.plan_final_eval import (
    BASELINES,
    CSS_EVALUATION_ITEMS,
    EIKOS_ARCHITECTURE,
    NativeModel,
    frozen_candidates,
    frozen_css_prompts,
    native_command,
    sha_file,
    shell,
    source_command,
)

PLAN_VERSION = "decision2-first-release-v3-plan/1"
ROSTER_VERSION = "decision2-first-release-v3-pretest-roster/1"
GATE_DOCUMENT = "research/jev-arena-v3-first-release-gates-2026-09-27.md"
SOURCE_FILES = (
    "benchmark/generate.py",
    "benchmark/score.py",
    "transfer/build.py",
    "transfer/score.py",
    "transfer/compare.py",
    "jev_arena/arena.py",
    "jev_arena/arena_v3.py",
    "jev_arena/compare_v3.py",
    "jev_arena/jevbench_public.py",
    "jev_arena/public_rank.py",
    "jev_arena/render.py",
    "publication/adapter_runtime.py",
    "publication/bundle_arena.py",
    "publication/generate_arena.py",
    "publication/generate_arena_v3.py",
    "publication/render_arena_v3.py",
    "publication/bundle_arena_v3.py",
    "scripts/plan_final_eval.py",
    "scripts/baseline_attestation_v3.py",
    "scripts/plan_first_release_v3.py",
    "scripts/freeze_first_release_v3.py",
)
SAME_SIZE_COMPARATOR = {
    "0.6B": {"kai", "lex"},
    "0.8B": {"eos"},
    "2B": {"sol"},
    "4B": {"nox"},
    "9B": {"lux"},
}
BASELINE_ROW_FILES = {
    "decider": {"model_config_sha256": "decider_config.json"},
    "eikos4b": {
        "model_config_sha256": "decision_config.json",
        "release_manifest_sha256": "SHA256SUMS",
    },
    "eos": {"model_config_sha256": "MODEL_MANIFEST.json"},
    "kai": {"model_config_sha256": "native/MANIFEST.json"},
    "laya": {
        "model_config_sha256": "rl_agent_config.json",
        "model_weights_sha256": "model.safetensors",
    },
    "lex": {"model_config_sha256": "native/MANIFEST.json"},
    "lux": {"model_config_sha256": "bundle-manifest.json"},
    "nox": {"model_config_sha256": "bundle-manifest.json"},
    "sol": {"model_config_sha256": "bundle-manifest.json"},
    "this-that": {
        "model_config_sha256": "config.json",
        "model_weights_sha256": "model.safetensors",
    },
    "jevk5-2b": {
        "model_config_sha256": "jevk5_config.json",
        "model_weight_sha256": "model.safetensors",
    },
    "jevk5-4b": {
        "model_config_sha256": "jevk5_config.json",
        "model_weight_sha256": "model.safetensors",
        "release_manifest_sha256": "SHA256SUMS",
    },
    "jevk5-9b": {
        "model_config_sha256": "jevk5_config.json",
        "model_weight_sha256": "model.safetensors",
        "release_manifest_sha256": "SHA256SUMS",
    },
}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def pair_digest(pairs: list[dict[str, str]]) -> str:
    """Canonical pretest identity of candidate-to-1.0 comparisons."""
    ordered = sorted(pairs, key=lambda pair: pair["candidate"])
    encoded = json.dumps(
        ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _abs_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    return Path(value)


def _source_hashes(
    source_root: Path, models: list[NativeModel | dict[str, Any]]
) -> dict[str, str]:
    names = set(SOURCE_FILES)
    for model in models:
        if isinstance(model, NativeModel):
            names.add(model.module.replace(".", "/") + ".py")
            if model.key == "jev":
                names.add("transfer/normalize_jev.py")
        elif model.get("architecture") == EIKOS_ARCHITECTURE:
            names.add("training/eikos/published_infer.py")
        else:
            names.update(("training/model/infer.py", "training/model/calibration.py"))
    return {name: sha_file(source_root / name) for name in sorted(names)}


def _baseline_attestations(
    roster: dict[str, Any],
    selected: list[NativeModel],
    source_root: Path,
    model_root: Path,
    external_root: Path,
) -> dict[str, dict[str, Any]]:
    """Bind native baseline adapter/package receipts, leaving HF files private."""
    declared = roster.get("baseline_attestations")
    if not isinstance(declared, list) or len(declared) != len(selected):
        raise ValueError("Every selected baseline needs one native attestation")
    by_key = {model.key: model for model in selected}
    attestations: dict[str, dict[str, Any]] = {}
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            raise ValueError("Malformed baseline attestation")
        key = item["key"]
        if key not in by_key or key in attestations:
            raise ValueError("Unknown or duplicate baseline attestation")
        model = by_key[key]
        verify_attestation(
            item,
            model,
            source_root=source_root,
            model_root=model_root,
            external_root=external_root,
        )
        attestations[key] = item
    return attestations


def _baseline_prediction_hashes(
    model: NativeModel, attestation: dict[str, Any]
) -> dict[str, str]:
    """Use only native row digests with an exact attested file or defined recipe."""
    receipt_path = Path(attestation["receipt_path"])
    if sha_file(receipt_path) != attestation["receipt_sha256"]:
        raise ValueError(f"{model.key}: baseline receipt changed during audit")
    receipt = _object(receipt_path)
    files = receipt["files"]
    expected = {}
    for field, name in BASELINE_ROW_FILES.get(model.key, {}).items():
        if name not in files:
            raise ValueError(f"{model.key}: attested package lacks {field} source")
        expected[field] = files[name]
    if model.key == "kev":
        # Kev's field is a digest of three file digests, not one file SHA.
        names = {
            "provenance": "provenance.json",
            "head": "head.pt",
            "adapter": "adapter_model.safetensors",
        }
        if any(name not in files for name in names.values()):
            raise ValueError("kev: attested package lacks fingerprint sources")
        payload = {field: files[name] for field, name in names.items()}
        expected["model_config_sha256"] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    if model.module == "inference.jevk5":
        runtime = receipt["runtime_files"]
        names = ("jevk5/runtime.py", "jevk5/prompt.py")
        if any(name not in runtime for name in names):
            raise ValueError(f"{model.key}: attested runtime lacks source files")
        payload = {name: runtime[name] for name in names}
        expected["runtime_source_sha256"] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    return expected


def checked_roster(
    path: Path,
    *,
    candidates: list[dict[str, Any]],
    source_root: Path,
    model_root: Path,
    external_root: Path,
) -> tuple[
    list[NativeModel | dict[str, Any]],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, float],
    str,
]:
    roster = _object(path)
    if roster.get("schema_version") != ROSTER_VERSION:
        raise ValueError("Unknown first-release v3 roster schema")
    if roster.get("candidate_keys") != [candidate["key"] for candidate in candidates]:
        raise ValueError("Candidate roster differs from frozen selected candidates")
    keys = roster.get("baseline_keys")
    catalog = {model.key: model for model in BASELINES}
    if (
        not isinstance(keys, list)
        or not keys
        or len(keys) != len(set(keys))
        or any(key not in catalog for key in keys)
    ):
        raise ValueError("Baseline roster must select unique pinned catalog keys")
    selected = [catalog[key] for key in keys]
    if not any(model.group == "open" for model in selected):
        raise ValueError("First-release roster needs an open-model control")
    sizes = roster.get("candidate_size_b")
    candidate_by_key = {entry["key"]: entry for entry in candidates}
    if not isinstance(sizes, dict) or set(sizes) != set(candidate_by_key):
        raise ValueError(
            "Actual loaded parameter count is required for every candidate"
        )
    if any(
        type(size) not in (int, float) or not 0 < size < 1000 for size in sizes.values()
    ):
        raise ValueError("Candidate loaded parameter counts must be positive")
    attestations = _baseline_attestations(
        roster, selected, source_root, model_root, external_root
    )
    pairs = roster.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(candidates):
        raise ValueError("Every candidate needs one predeclared 1.0 comparator")
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {
            "candidate",
            "comparator",
            "size_relation",
            "rationale",
        }:
            raise ValueError("Comparison pair has missing or unknown fields")
        candidate_key, baseline_key = pair["candidate"], pair["comparator"]
        if candidate_key not in candidate_by_key or candidate_key in seen:
            raise ValueError("Duplicate or unknown comparison candidate")
        if baseline_key not in keys or catalog[baseline_key].group != "decision1":
            raise ValueError("Comparator must be a selected Decision 1.0 model")
        size = candidate_by_key[candidate_key]["size"]
        candidate_size = sizes[candidate_key]
        comparator_size = attestations[baseline_key]["size_b"]
        size_ratio = max(candidate_size, comparator_size) / min(
            candidate_size, comparator_size
        )
        same = (
            baseline_key in SAME_SIZE_COMPARATOR.get(size, set()) and size_ratio <= 1.25
        )
        if (same and pair["size_relation"] != "same") or (
            not same and pair["size_relation"] != "nearest"
        ):
            raise ValueError("Comparator size relation is false")
        if not same and (
            not isinstance(pair["rationale"], str)
            or len(pair["rationale"].strip()) < 20
        ):
            raise ValueError("Nearest-size comparison requires a disclosed rationale")
        seen.add(candidate_key)
    gate_hash = _sha(roster.get("gate_document_sha256"), "gate_document_sha256")
    if sha_file(source_root / GATE_DOCUMENT) != gate_hash:
        raise ValueError(
            "Numeric release gate document changed after pretest declaration"
        )
    return [*selected, *candidates], pairs, attestations, sizes, sha_file(path)


def build_plan(
    *,
    models: list[NativeModel | dict[str, Any]],
    pairs: list[dict[str, str]],
    attestations: dict[str, dict[str, Any]],
    candidate_freeze_path: Path,
    candidate_freeze_sha256: str,
    candidate_size_b: dict[str, float],
    roster_path: Path,
    roster_sha256: str,
    css_prompts: Path,
    css_info: dict[str, Any],
    public_panel: Path,
    source_root: Path,
    evaluation_root: Path,
    model_root: Path,
    external_root: Path,
    python: str,
    kai_lex_python: str,
    fla_path: str,
) -> dict[str, Any]:
    if evaluation_root.exists():
        raise FileExistsError("Evaluation root must be absent for a fresh v3 run")
    manifest = _object(public_panel / "manifest.json")
    if (
        manifest.get("build_version") != PUBLIC_BUILD_VERSION
        or manifest.get("items") != 231
        or manifest.get("source_url") != PUBLIC_SOURCE_URL
        or manifest.get("source_revision") != PUBLIC_SOURCE_REVISION
        or manifest.get("source_sha256")
        != {tier: digest for tier, (_, digest, _) in PUBLIC_FILES.items()}
    ):
        raise ValueError("Expected a pinned JevBench public 231-item build")
    for field in ("prompts_sha256", "targets_sha256"):
        _sha(manifest.get(field), f"public.{field}")
    public_manifest_sha = sha_file(public_panel / "manifest.json")
    source_hashes = _source_hashes(source_root, models)
    final_prompts = evaluation_root / "typed-final.prompts.jsonl"
    final_gold = evaluation_root / "typed-final.gold.jsonl"
    css_gold = css_prompts.with_name("css-evaluation.gold.jsonl")
    public_prompts = public_panel / "prompts.jsonl"
    predictions = []
    scoring = []
    roster_models = []
    for model in models:
        candidate = isinstance(model, dict)
        key = model["key"] if candidate else model.key
        model_id = model["model_id"] if candidate else model.model_id
        revision = model["selected_checkpoint"] if candidate else model.revision
        group = "decision2" if candidate else model.group
        pred = {
            panel: evaluation_root / "predictions" / f"{key}.{panel}.predictions.jsonl"
            for panel in ("typed", "css", "public")
        }
        commands = []
        for panel, prompt in (
            ("typed", final_prompts),
            ("css", css_prompts),
            ("public", public_prompts),
        ):
            commands.extend(
                native_command(
                    model,
                    prompts=prompt,
                    output=pred[panel],
                    model_root=model_root,
                    external_root=external_root,
                    source_root=source_root,
                    python=python,
                    kai_lex_python=kai_lex_python,
                    fla_path=fla_path,
                )
            )
        predictions.append(
            {
                "key": key,
                "model_id": model_id,
                "revision": revision,
                "group": group,
                "paths": {panel: str(path) for panel, path in pred.items()},
                "commands": commands,
            }
        )
        typed_report = evaluation_root / "reports" / f"{key}.typed.score.json"
        css_report = evaluation_root / "reports" / f"{key}.css.score.json"
        public_report = evaluation_root / "reports" / f"{key}.public.score.json"
        backend = (
            "eikos-semif-native"
            if candidate and model.get("architecture") == EIKOS_ARCHITECTURE
            else "decision2-native-calibrated" if candidate else model.backend
        )
        public_command = [
            python,
            "-m",
            "jev_arena.jevbench_public",
            "score",
            "--panel-dir",
            public_panel,
            "--predictions",
            pred["public"],
            "--model-id",
            model_id,
            "--model-revision",
            revision,
            "--output",
            public_report,
        ]
        if candidate:
            public_command += [
                "--prediction-manifest",
                Path(str(pred["public"]) + ".manifest.json"),
            ]
        scoring.append(
            {
                "key": key,
                "typed_report": str(typed_report),
                "css_report": str(css_report),
                "public_report": str(public_report),
                "commands": [
                    source_command(
                        source_root,
                        python,
                        "-m",
                        "benchmark.score",
                        "--gold",
                        final_gold,
                        "--predictions",
                        pred["typed"],
                        "--model-id",
                        model_id,
                        "--model-revision",
                        revision,
                        "--backend",
                        backend,
                        "--output",
                        typed_report,
                    ),
                    source_command(
                        source_root,
                        python,
                        "-m",
                        "transfer.score",
                        "--gold",
                        css_gold,
                        "--predictions",
                        pred["css"],
                        "--output",
                        css_report,
                    ),
                    source_command(source_root, *public_command),
                ],
            }
        )
        identity = (
            model["model_sha256"]
            if candidate
            else attestations[key]["native_model_sha256"]
        )
        adapter = (
            source_hashes["training/eikos/published_infer.py"]
            if candidate and model.get("architecture") == EIKOS_ARCHITECTURE
            else (
                source_hashes["training/model/infer.py"]
                if candidate
                else attestations[key]["adapter_sha256"]
            )
        )
        calibration = (
            model["calibration_sha256"]
            if candidate
            else attestations[key]["calibration_sha256"]
        )
        roster_models.append(
            {
                "key": key,
                "label": model["label"] if candidate else model.label,
                "group": group,
                "model_id": model_id,
                "revision": revision,
                "native_model_sha256": identity,
                "adapter_sha256": adapter,
                "calibration_sha256": calibration,
                "size_b": (
                    candidate_size_b[key] if candidate else attestations[key]["size_b"]
                ),
                "typed_report": str(typed_report),
                "css_report": str(css_report),
                "public_report": str(public_report),
            }
        )
    comparisons = []
    for pair in pairs:
        left, right = pair["candidate"], pair["comparator"]
        pred = {model["key"]: model for model in predictions}
        output = evaluation_root / "comparisons" / f"{left}-vs-{right}.v3.paired.json"
        comparisons.append(
            {
                **pair,
                "output": str(output),
                "command": source_command(
                    source_root,
                    python,
                    "-m",
                    "jev_arena.compare_v3",
                    "--typed-gold",
                    final_gold,
                    "--css-gold",
                    css_gold,
                    "--left-typed",
                    pred[left]["paths"]["typed"],
                    "--left-css",
                    pred[left]["paths"]["css"],
                    "--right-typed",
                    pred[right]["paths"]["typed"],
                    "--right-css",
                    pred[right]["paths"]["css"],
                    "--left-name",
                    pred[left]["model_id"],
                    "--right-name",
                    pred[right]["model_id"],
                    "--replicates",
                    "5000",
                    "--seed",
                    "20260927",
                    "--output",
                    output,
                ),
            }
        )
    raw_paths = []
    for item in predictions:
        for path in item["paths"].values():
            raw_paths.append(Path(path))
            if item["group"] == "decision2":
                raw_paths.append(Path(path + ".manifest.json"))
    arena_roster_path = evaluation_root / "reports" / "arena-v3-roster.json"
    arena_rank_path = evaluation_root / "reports" / "arena-v3-rank.json"
    public_roster_path = evaluation_root / "reports" / "jevbench-public-roster.json"
    public_rank_path = evaluation_root / "reports" / "jevbench-public-rank.json"
    arena_entries = [
        {
            field: row[field]
            for field in (
                "key",
                "label",
                "group",
                "model_id",
                "revision",
                "size_b",
                "typed_report",
                "css_report",
            )
        }
        for row in roster_models
    ]
    public_entries = [
        {
            field: row[field]
            for field in (
                "key",
                "label",
                "group",
                "model_id",
                "revision",
                "size_b",
            )
        }
        | {"report": row["public_report"]}
        for row in roster_models
    ]
    return {
        "plan_version": PLAN_VERSION,
        "status": "commands_only_not_executed",
        "panel_policy": "JevArena v3 sealed 1600 typed + 6547 CSS; JevBench public231 separate; authored v3.1 deferred",
        "formula": "100*sqrt(T*H)",
        "candidate_freeze_sha256": candidate_freeze_sha256,
        "candidate_freeze_path": str(candidate_freeze_path),
        "pretest_roster_path": str(roster_path),
        "pretest_roster_sha256": roster_sha256,
        "comparison_pairs_sha256": pair_digest(pairs),
        "comparison_pairs": pairs,
        "gate_document_sha256": sha_file(source_root / GATE_DOCUMENT),
        "css_prompts": {"path": str(css_prompts), **css_info},
        "public_panel": {
            "path": str(public_panel),
            "manifest_sha256": public_manifest_sha,
            "prompts_sha256": manifest["prompts_sha256"],
            "targets_sha256": manifest["targets_sha256"],
            "items": 231,
        },
        "source_root": str(source_root),
        "model_root": str(model_root),
        "external_root": str(external_root),
        "evaluation_root": str(evaluation_root),
        "source_sha256": source_hashes,
        "model_roster": roster_models,
        "preparation_commands_after_candidate_freeze": [
            shell("install", "-d", "-m", "700", evaluation_root),
            shell(
                "install",
                "-d",
                "-m",
                "700",
                evaluation_root / "predictions",
                evaluation_root / "reports",
                evaluation_root / "comparisons",
            ),
            shell(
                python,
                "-c",
                "import os,secrets,sys; fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(fd,secrets.token_bytes(32)); os.close(fd)",
                evaluation_root / "typed-final.seed",
            ),
            source_command(
                source_root,
                python,
                "-m",
                "benchmark.generate",
                "--split",
                "final",
                "--seed-file",
                evaluation_root / "typed-final.seed",
                "--groups-per-family",
                "100",
                "--output",
                final_gold,
                "--prompts-output",
                final_prompts,
            ),
        ],
        "inference": predictions,
        "raw_prediction_hash_command": "set -C; "
        + shell("sha256sum", *raw_paths)
        + " > "
        + shlex.quote(str(evaluation_root / "RAW_PREDICTIONS.sha256")),
        "scoring_commands_after_prekey_freeze": scoring,
        "paired_ci_commands_after_prekey_freeze": comparisons,
        "rank_input_templates_after_prekey_freeze": {
            "arena_roster_path": str(arena_roster_path),
            "arena_roster": {
                "schema_version": "jevarena-v3-roster/1",
                "phase": "release",
                "freeze_receipt": str(
                    evaluation_root / "reports" / "arena-v3-prekey-freeze.json"
                ),
                "freeze_sha256": "REPLACE_WITH_ACTUAL_PREKEY_FREEZE_SHA256",
                "models": arena_entries,
            },
            "public_roster_path": str(public_roster_path),
            "public_roster": {"models": public_entries},
        },
        "rank_commands_after_reports": [
            source_command(
                source_root,
                python,
                "-m",
                "jev_arena.arena_v3",
                "--manifest",
                arena_roster_path,
                "--output",
                arena_rank_path,
            ),
            source_command(
                source_root,
                python,
                "-m",
                "jev_arena.public_rank",
                "--manifest",
                public_roster_path,
                "--output",
                public_rank_path,
            ),
        ],
        "required_report_versions": {
            "typed": "typed-decision-report/2",
            "css": "css-transfer-score/2",
            "arena": "jevarena-ranking/3",
            "paired": "jevarena-v3-paired-aggregate/1",
            "jevbench_public": "jevarena-jevbench-public-score/1",
        },
        "prekey_freeze_requires": [
            "Lock the candidate roster, comparator pairs, policy and code before generating predictions.",
            "Hash and verify all 8147 gold-free prediction rows before reading either held-out answer file.",
            "After prediction seals and audit, bind typed/CSS gold digests, plan/audit receipts, all prediction digests, package and adapter identities, exact source hashes, numeric policy and formula in jevarena-v3-freeze/2.",
            "Seal the v3 pre-key receipt and its SHA before first label access; record candidate-lock, prediction-seal, receipt-seal and label-access times in an independently reviewed event log.",
            "JevBench public231 is separately scored and required for release; it never changes T, H or the v3 rank.",
        ],
        "release_integration_requires": [
            "The pre-key v3 freeze and publication gate must verify comparison_pairs_sha256 and each candidate-to-1.0 mapping.",
            "A matching plan and gold-free audit are necessary evidence, never an automatic publication pass.",
        ],
    }


def audit_prekey_predictions(plan: dict[str, Any]) -> dict[str, Any]:
    """Gold-free row, source and identity audit after inference, before scoring."""
    source_root = Path(plan["source_root"])
    candidate_freeze = Path(plan["candidate_freeze_path"])
    if sha_file(candidate_freeze) != plan["candidate_freeze_sha256"]:
        raise ValueError("Candidate lock changed since v3 planning")
    frozen, _ = frozen_candidates(candidate_freeze)
    if {row["key"] for row in frozen} != {
        row["key"] for row in plan["model_roster"] if row["group"] == "decision2"
    }:
        raise ValueError("Candidate package roster changed since v3 planning")
    models, pairs, attestations, sizes, _ = checked_roster(
        Path(plan["pretest_roster_path"]),
        candidates=frozen,
        source_root=source_root,
        model_root=Path(plan["model_root"]),
        external_root=Path(plan["external_root"]),
    )
    if len(plan["model_roster"]) != len(models) or {
        model.key if isinstance(model, NativeModel) else model["key"]
        for model in models
    } != {row["key"] for row in plan["model_roster"]}:
        raise ValueError("Native model roster changed since v3 planning")
    if (
        pair_digest(pairs) != plan["comparison_pairs_sha256"]
        or pairs != plan["comparison_pairs"]
    ):
        raise ValueError("Predeclared 1.0 comparison pairs changed since v3 planning")
    baseline_by_key = {
        model.key: model for model in models if isinstance(model, NativeModel)
    }
    baseline_row_hashes = {
        key: _baseline_prediction_hashes(model, attestations[key])
        for key, model in baseline_by_key.items()
    }
    for row in plan["model_roster"]:
        key = row["key"]
        expected = (
            sizes[key] if row["group"] == "decision2" else attestations[key]["size_b"]
        )
        if row["size_b"] != expected:
            raise ValueError(f"{key}: loaded parameter count changed")
        if key in baseline_by_key:
            baseline = baseline_by_key[key]
            attestation = attestations[key]
            expected_identity = {
                "group": baseline.group,
                "model_id": baseline.model_id,
                "revision": baseline.revision,
                "native_model_sha256": attestation["native_model_sha256"],
                "adapter_sha256": attestation["adapter_sha256"],
                "calibration_sha256": attestation["calibration_sha256"],
            }
            if any(
                row.get(field) != value for field, value in expected_identity.items()
            ):
                raise ValueError(
                    f"{key}: planned baseline identity differs from attestation"
                )
    if any(
        sha_file(source_root / name) != digest
        for name, digest in plan["source_sha256"].items()
    ):
        raise ValueError("Protocol source changed since v3 planning")
    if sha_file(source_root / GATE_DOCUMENT) != plan["gate_document_sha256"]:
        raise ValueError("Numeric gate document changed since v3 planning")
    if sha_file(Path(plan["pretest_roster_path"])) != plan["pretest_roster_sha256"]:
        raise ValueError("Pretest roster changed since v3 planning")
    if sha_file(Path(plan["css_prompts"]["path"])) != plan["css_prompts"]["sha256"]:
        raise ValueError("CSS gold-free prompts changed")
    if (
        sha_file(Path(plan["public_panel"]["path"]) / "manifest.json")
        != plan["public_panel"]["manifest_sha256"]
    ):
        raise ValueError("Public panel manifest changed")
    panel_paths = {
        "typed": Path(plan["evaluation_root"]) / "typed-final.prompts.jsonl",
        "css": Path(plan["css_prompts"]["path"]),
        "public": Path(plan["public_panel"]["path"]) / "prompts.jsonl",
    }
    prompt_inputs: dict[str, dict[str, tuple[str, set[str]]]] = {}
    prompt_sha: dict[str, str] = {}
    for panel, path in panel_paths.items():
        inputs: dict[str, tuple[str, set[str]]] = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or set(row) != {"id", "state", "questions"}
                    or not isinstance(row.get("id"), str)
                    or row["id"] in inputs
                    or not isinstance(row.get("questions"), dict)
                    or not row["questions"]
                ):
                    raise ValueError(
                        f"{panel}: malformed or duplicate gold-free prompt ID"
                    )
                payload = {"state": row["state"], "questions": row["questions"]}
                encoded = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                inputs[row["id"]] = (
                    hashlib.sha256(encoded.encode()).hexdigest(),
                    set(row["questions"]),
                )
        expected = {"typed": 1600, "css": CSS_EVALUATION_ITEMS, "public": 231}[panel]
        if len(inputs) != expected:
            raise ValueError(f"{panel}: expected {expected} gold-free prompt IDs")
        prompt_inputs[panel], prompt_sha[panel] = inputs, sha_file(path)
    if prompt_sha["public"] != plan["public_panel"]["prompts_sha256"]:
        raise ValueError("Public gold-free prompts changed")
    hash_lines = []
    frozen_by_key = {row["key"]: row for row in frozen}
    for model in plan["inference"]:
        model_hashes = {}
        for panel, name in model["paths"].items():
            path = Path(name)
            seen = set()
            with path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or not isinstance(row.get("id"), str)
                        or row["id"] in seen
                    ):
                        raise ValueError(
                            f"{model['key']} {panel}:{number}: bad prediction ID"
                        )
                    if set(row) & {
                        "gold",
                        "target_probs",
                        "teacher_probs",
                        "provenance",
                    }:
                        raise ValueError("Prediction contains answer-only fields")
                    expected_input = prompt_inputs[panel].get(row["id"])
                    if (
                        expected_input is None
                        or row.get("source_input_sha256") != expected_input[0]
                        or not isinstance(row.get("answers"), dict)
                        or set(row["answers"]) != expected_input[1]
                    ):
                        raise ValueError(
                            f"{model['key']} {panel}:{number}: input or answer IDs differ"
                        )
                    if (
                        row.get("model_id", model["model_id"]) != model["model_id"]
                        or row.get("model_revision") != model["revision"]
                    ):
                        raise ValueError(
                            f"{model['key']} {panel}:{number}: model identity differs"
                        )
                    if model["group"] == "decision2":
                        identity = frozen_by_key[model["key"]]
                        if (
                            row.get("model_sha256") != identity["model_sha256"]
                            or row.get("calibration_sha256")
                            != identity["calibration_sha256"]
                        ):
                            raise ValueError(
                                f"{model['key']} {panel}:{number}: package or CAL differs"
                            )
                    else:
                        baseline = baseline_by_key[model["key"]]
                        if (
                            row.get("backend") != baseline.backend
                            or row.get("revision_attested") is not True
                            or any(
                                row.get(field) != digest
                                for field, digest in baseline_row_hashes[
                                    model["key"]
                                ].items()
                            )
                            or (
                                baseline.group == "decision1"
                                and row.get("runtime_matches_validated") is not True
                            )
                        ):
                            raise ValueError(
                                f"{model['key']} {panel}:{number}: native runtime differs"
                            )
                    seen.add(row["id"])
            if seen != set(prompt_inputs[panel]):
                raise ValueError(
                    f"{model['key']} {panel}: prediction IDs differ from prompts"
                )
            model_hashes[panel] = sha_file(path)
            hash_lines.append(f"{model_hashes[panel]}  {path}")
            if model["group"] == "decision2":
                manifest = Path(name + ".manifest.json")
                if not manifest.is_file():
                    raise ValueError(
                        f"{model['key']} {panel}: missing native prediction manifest"
                    )
                receipt = _object(manifest)
                identity = frozen_by_key[model["key"]]
                if (
                    receipt.get("model_id") != model["model_id"]
                    or receipt.get("model_revision") != model["revision"]
                    or receipt.get("model_sha256") != identity["model_sha256"]
                    or receipt.get("input_sha256") != prompt_sha[panel]
                    or receipt.get("input_items") != len(prompt_inputs[panel])
                    or receipt.get("predictions_sha256") != model_hashes[panel]
                    or receipt.get("calibration", {}).get("file_sha256")
                    != identity["calibration_sha256"]
                ):
                    raise ValueError(
                        f"{model['key']} {panel}: native manifest binding differs"
                    )
                hash_lines.append(f"{sha_file(manifest)}  {manifest}")
        model["audited_prediction_sha256"] = model_hashes
    raw = Path(plan["evaluation_root"]) / "RAW_PREDICTIONS.sha256"
    if raw.read_text(encoding="utf-8").splitlines() != hash_lines:
        raise ValueError("Saved raw prediction SHA256 list differs from audited bytes")
    return {
        "status": "gold_free_prekey_predictions_verified",
        "prompt_sha256": prompt_sha,
        "models": {
            row["key"]: row["audited_prediction_sha256"] for row in plan["inference"]
        },
        "raw_hashes_sha256": sha_file(raw),
        "comparison_pairs_sha256": plan["comparison_pairs_sha256"],
        "note": "No FINAL labels were opened by this auditor",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    plan_parser = actions.add_parser(
        "plan", help="Validate pretest freeze and print commands only"
    )
    plan_parser.add_argument("--candidate-freeze", type=Path, required=True)
    plan_parser.add_argument("--roster", type=Path, required=True)
    plan_parser.add_argument("--css-prompts", type=Path, required=True)
    plan_parser.add_argument("--public-panel", type=Path, required=True)
    plan_parser.add_argument("--source-root", type=Path, required=True)
    plan_parser.add_argument("--evaluation-root", type=Path, required=True)
    plan_parser.add_argument("--model-root", type=Path, required=True)
    plan_parser.add_argument("--external-root", type=Path, required=True)
    plan_parser.add_argument("--python", default="python3")
    plan_parser.add_argument(
        "--kai-lex-python", default="/work/envs/kai-lex/bin/python"
    )
    plan_parser.add_argument("--fla-path", default="/opt/decision-fla")
    audit_parser = actions.add_parser(
        "audit-predictions", help="Audit pre-key gold-free prediction files"
    )
    audit_parser.add_argument("--plan", type=Path, required=True)
    audit_parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    if args.action == "audit-predictions":
        _abs_path(str(args.plan), "plan")
        if sha_file(args.plan) != _sha(
            args.expected_plan_sha256, "expected_plan_sha256"
        ):
            raise ValueError("Saved plan differs from its predeclared SHA-256")
        plan = _object(args.plan)
        if plan.get("plan_version") != PLAN_VERSION:
            raise ValueError("Unknown saved v3 plan")
        result = audit_prekey_predictions(plan)
    else:
        for name in (
            "candidate_freeze",
            "roster",
            "css_prompts",
            "public_panel",
            "source_root",
            "evaluation_root",
            "model_root",
            "external_root",
        ):
            _abs_path(str(getattr(args, name)), name)
        for name in ("kai_lex_python", "fla_path"):
            _abs_path(getattr(args, name), name)
        candidates, candidate_sha = frozen_candidates(args.candidate_freeze)
        css = frozen_css_prompts(args.css_prompts)
        models, pairs, attestations, sizes, roster_sha = checked_roster(
            args.roster,
            candidates=candidates,
            source_root=args.source_root,
            model_root=args.model_root,
            external_root=args.external_root,
        )
        result = build_plan(
            models=models,
            pairs=pairs,
            attestations=attestations,
            candidate_freeze_path=args.candidate_freeze,
            candidate_freeze_sha256=candidate_sha,
            candidate_size_b=sizes,
            roster_path=args.roster,
            roster_sha256=roster_sha,
            css_prompts=args.css_prompts,
            css_info=css,
            public_panel=args.public_panel,
            source_root=args.source_root,
            evaluation_root=args.evaluation_root,
            model_root=args.model_root,
            external_root=args.external_root,
            python=args.python,
            kai_lex_python=args.kai_lex_python,
            fla_path=args.fla_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
