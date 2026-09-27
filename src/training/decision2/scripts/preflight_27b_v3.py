"""Gold-free first-release lock for the official-Qwen-start 27B candidate.

This module never reads evaluation labels, scores predictions, or starts GPU
inference. It binds the exact uppercase PEFT package, its official Qwen source,
qualified comparators, prompt bytes and current v3 scoring code. A separate
command seals the candidate's three complete package-native prediction files.
Neither receipt is a release decision or a claim that v3 is virgin blind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_arena.arena_v3 import SCORER_SOURCE_PATHS
from inference.autojev27 import (
    MODEL_ID as AUTOJEV_ID,
    MODEL_REVISION as AUTOJEV_REVISION,
)
from inference.autojev27 import verify_release as verify_autojev_release
from publication.adapter_bundle import _verify_staged_runtime
from publication.adapter_runtime import verify_bundle
from publication.bundle_arena import _parity
from publication.package_native_arena import (
    ADAPTER_VERSION,
    input_digest,
    load_gold_free,
)
from scripts.plan_final_eval import BASELINES, NativeModel, sha_file
from scripts.plan_first_release_v3 import (
    _baseline_attestations,
    _baseline_repeatability,
)
from training.model.data import canonical
from training.model.infer import checkpoint_fingerprint

LOCK_VERSION = "decision2-27b-v3-candidate-lock/1"
SEAL_VERSION = "decision2-27b-v3-prediction-seal/1"
MODEL_ID = "llm-semantic-router/DEV2.0-27B"
MODEL_KEY = "d2-27b"
BASELINE_KEYS = ("lux", "jevk5-9b", "autojev27b")
AUTOJEV_MODEL = NativeModel(
    "autojev27b",
    "AutoJev 27B",
    "open",
    "27B",
    AUTOJEV_ID,
    AUTOJEV_REVISION,
    "autojev-native-eager",
    "inference.autojev27",
    "autojev-27b",
    "autojev-source",
    qualification="Pinned native three-type DecisionModel; 8192-token overflow invalid.",
)
BASE_ID = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BEST368_SHA256 = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
SCORED_MANIFEST_SHA256 = (
    "b2e33b1a4daa410ab6ab1d402692b296298fb29defa771c8c5cf6344e84b0760"
)
CAL_SHA256 = "e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78"
LOADED_PARAMETERS = 25_688_227_840
PANELS = {"typed": 1600, "css": 6547, "public": 231}
QUALIFICATION_SOURCES = (
    "scripts/plan_final_eval.py",
    "scripts/plan_first_release_v3.py",
    "scripts/baseline_attestation_v3.py",
    "scripts/baseline_repeat_smoke_v3.py",
    "scripts/attest_autojev27_v3.py",
    "publication/adapter_bundle.py",
    "publication/adapter_runtime.py",
    "publication/panel_parity.py",
    "inference/autojev27.py",
)
POLICY = (
    Path(__file__).resolve().parents[1]
    / "research/decision2-27b-v3-first-release-policy-2026-09-27.md"
)
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("A required private JSON file is missing or linked")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("A required private JSON file is not an object")
    return value


def _private_write(path: Path, value: dict[str, Any], source_root: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise ValueError("Receipt output must be a new absolute private file")
    parent = path.parent.resolve(strict=True)
    if parent.is_relative_to(source_root.resolve(strict=True)):
        raise ValueError("Private receipt cannot be written into source control")
    if (
        not parent.is_dir()
        or stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
    ):
        raise ValueError("Receipt parent must be owner-held mode 0700")
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _prompts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    if set(paths) != set(PANELS):
        raise ValueError("Typed FINAL, CSS15 and public231 prompts are all required")
    result = {}
    for name, expected in PANELS.items():
        path = paths[name]
        if not path.is_absolute():
            raise ValueError("Prompt paths must be absolute")
        rows = load_gold_free(path)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} gold-free prompt items")
        result[name] = {
            "path": str(path),
            "sha256": sha_file(path),
            "items": expected,
            "question_count": sum(len(row["questions"]) for row in rows),
        }
    if result["typed"]["question_count"] != 2000:
        raise ValueError("Typed FINAL must contain 2,000 scored questions")
    if result["css"]["question_count"] != 6547:
        raise ValueError("CSS15 must contain one question per item")
    return result


def _source_hashes(source_root: Path) -> dict[str, str]:
    sources = {name: sha_file(path) for name, path in SCORER_SOURCE_PATHS.items()}
    sources["collector"] = sha_file(source_root / "publication/package_native_arena.py")
    sources["preflight"] = sha_file(Path(__file__))
    sources.update(
        {
            f"qualification/{name}": sha_file(source_root / name)
            for name in QUALIFICATION_SOURCES
        }
    )
    return sources


def _comparators(
    roster: dict[str, Any],
    source_root: Path,
    model_root: Path,
    external_root: Path,
    candidate_size_b: float,
) -> dict[str, Any]:
    keys = roster.get("baseline_keys")
    catalog = {model.key: model for model in BASELINES}
    if keys != list(BASELINE_KEYS) or any(key not in catalog for key in keys[:2]):
        raise ValueError("27B roster must pin Lux, JevK5 and AutoJev before inference")
    selected = [catalog[key] for key in keys[:2]] + [AUTOJEV_MODEL]
    if (
        selected[0].group != "decision1"
        or selected[1].group != "open"
        or selected[2].group != "open"
    ):
        raise ValueError("27B comparator catalog groups changed")
    jebadiah = roster.get("jebadiah27b")
    if (
        not isinstance(jebadiah, dict)
        or set(jebadiah) != {"model_id", "status", "reason"}
        or jebadiah["model_id"] != "frontier-infra/jebadiah-27b"
        or jebadiah["status"] != "hold_unqualified"
        or not isinstance(jebadiah["reason"], str)
        or len(jebadiah["reason"].strip()) < 20
    ):
        raise ValueError("Jebadiah pre-inference qualification decision is missing")
    autojev_release = verify_autojev_release(
        model_root / AUTOJEV_MODEL.model_dir,
        external_root / AUTOJEV_MODEL.source_dir,
        AUTOJEV_REVISION,
    )
    attestations = _baseline_attestations(
        roster, selected, source_root, model_root, external_root
    )
    if (
        attestations["autojev27b"]["native_model_sha256"]
        != autojev_release["native_model_sha256"]
        or round(attestations["autojev27b"]["size_b"] * 1_000_000_000)
        != autojev_release["loaded_parameters"]
    ):
        raise ValueError("AutoJev native release differs from full package attestation")
    repeats = _baseline_repeatability(
        roster,
        selected,
        attestations,
        max_probability_drift_by_key={"autojev27b": 0.02},
    )
    for item in roster["baseline_repeatability"]:
        if item["key"] != "autojev27b":
            continue
        for field in ("first_predictions_path", "second_predictions_path"):
            with Path(item[field]).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if (
                        row.get("native_model_sha256")
                        != autojev_release["native_model_sha256"]
                        or row.get("runtime_source_sha256")
                        != autojev_release["runtime_source_sha256"]
                        or row.get("model_config_sha256")
                        != autojev_release["model_config_sha256"]
                    ):
                        raise ValueError(
                            "AutoJev smoke row differs from attested weights"
                        )
    pair = roster.get("decision1_pair")
    if (
        not isinstance(pair, dict)
        or set(pair) != {"comparator", "size_relation", "rationale"}
        or pair["comparator"] != "lux"
        or pair["size_relation"] != "nearest"
        or not isinstance(pair["rationale"], str)
        or len(pair["rationale"].strip()) < 20
    ):
        raise ValueError("Lux is a nearest-size comparator, never a same-size claim")
    open_pair = roster.get("open_control")
    if (
        not isinstance(open_pair, dict)
        or set(open_pair) != {"key", "size_relation", "rationale"}
        or open_pair["key"] != "jevk5-9b"
    ):
        raise ValueError("Open control must be a qualified catalog model")
    open_size = attestations[open_pair["key"]]["size_b"]
    ratio = max(candidate_size_b, open_size) / min(candidate_size_b, open_size)
    relation = "same" if ratio <= 1.25 else "nearest"
    if open_pair["size_relation"] != relation:
        raise ValueError("Open-control size relation differs from loaded parameters")
    if relation == "nearest" and (
        not isinstance(open_pair["rationale"], str)
        or len(open_pair["rationale"].strip()) < 20
    ):
        raise ValueError("A smaller open control requires an explicit size caveat")
    autojev_pair = roster.get("same_size_control")
    if (
        not isinstance(autojev_pair, dict)
        or set(autojev_pair) != {"key", "size_relation", "rationale"}
        or autojev_pair["key"] != "autojev27b"
    ):
        raise ValueError("AutoJev same-size comparison must be predeclared")
    autojev_size = attestations["autojev27b"]["size_b"]
    autojev_ratio = max(candidate_size_b, autojev_size) / min(
        candidate_size_b, autojev_size
    )
    if autojev_ratio > 1.25 or autojev_pair["size_relation"] != "same":
        raise ValueError("AutoJev loaded size is not comparable to candidate 27B")
    return {
        "baseline_keys": keys,
        "baseline_models": {
            model.key: {
                "model_id": model.model_id,
                "revision": model.revision,
                "native_model_sha256": attestations[model.key]["native_model_sha256"],
                "adapter_sha256": attestations[model.key]["adapter_sha256"],
                "calibration_sha256": attestations[model.key]["calibration_sha256"],
                "loaded_parameters": round(
                    attestations[model.key]["size_b"] * 1_000_000_000
                ),
            }
            for model in selected
        },
        "decision1_pair": pair,
        "open_control": open_pair,
        "same_size_control": autojev_pair,
        "jebadiah27b": jebadiah,
        "measured_size_ratio_to_lux": candidate_size_b / attestations["lux"]["size_b"],
        "measured_size_ratio_to_open": ratio,
        "measured_size_ratio_to_autojev": autojev_ratio,
        "attestation_sha256": {
            key: item["receipt_sha256"] for key, item in attestations.items()
        },
        "repeatability_sha256": {
            key: item["receipt_sha256"] for key, item in repeats.items()
        },
    }


def candidate_lock(
    *,
    package: Path,
    source: Path,
    expected_package_sha256: str,
    parity_receipt: Path,
    parity_sha256: str,
    roster_path: Path,
    prompt_paths: dict[str, Path],
    prediction_dir: Path,
    source_root: Path,
    model_root: Path,
    external_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate immutable candidate inputs and write one private pre-inference lock."""
    if any(
        not path.is_absolute()
        for path in (
            package,
            source,
            parity_receipt,
            roster_path,
            prediction_dir,
            source_root,
            model_root,
            external_root,
            output,
        )
    ):
        raise ValueError("All candidate lock paths must be absolute")
    expected = _sha(expected_package_sha256, "package manifest")
    if sha_file(package / "MODEL_MANIFEST.json") != expected:
        raise ValueError("Uppercase package manifest differs from expected bytes")
    manifest = verify_bundle(package)
    # This checks every packaged file, every external Qwen file, the PEFT
    # checkpoint fingerprint, CAL and loaded parameter inventory on CPU.
    _verify_staged_runtime(package, source)
    identity = checkpoint_fingerprint(package / "model", source)
    if (
        manifest.get("model_id") != MODEL_ID
        or manifest.get("base", {}).get("repo_id") != BASE_ID
        or manifest["base"].get("revision") != BASE_REVISION
        or manifest.get("model_sha256") != BEST368_SHA256
        or manifest.get("scored_prediction_manifest_sha256") != SCORED_MANIFEST_SHA256
        or identity.get("model_sha256") != BEST368_SHA256
        or manifest.get("calibration_sha256") != CAL_SHA256
        or manifest.get("parameter_count") != LOADED_PARAMETERS
        or manifest.get("publication_status") != "candidate-parity-pending"
    ):
        raise ValueError(
            "27B package lineage, native identity or parameter count differs"
        )
    if sha_file(parity_receipt) != _sha(parity_sha256, "parity receipt"):
        raise ValueError("Native parity receipt changed")
    revision = f"package-sha256:{expected}"
    _parity(
        _object(parity_receipt),
        MODEL_ID,
        revision,
        BEST368_SHA256,
        hashlib.sha256(canonical(identity["files_sha256"]).encode()).hexdigest(),
        CAL_SHA256,
    )
    roster = _object(roster_path)
    comparisons = _comparators(
        roster,
        source_root,
        model_root,
        external_root,
        LOADED_PARAMETERS / 1_000_000_000,
    )
    prompts = _prompts(prompt_paths)
    declared_prompts = roster.get("panel_prompt_sha256")
    if (
        not isinstance(declared_prompts, dict)
        or set(declared_prompts) != set(PANELS)
        or any(
            _sha(declared_prompts[name], name) != prompts[name]["sha256"]
            for name in PANELS
        )
    ):
        raise ValueError(
            "Gold-free prompt bytes differ from the predeclared panel roster"
        )
    if prediction_dir.exists() or prediction_dir.is_symlink():
        raise FileExistsError("Prediction directory must be fresh at candidate lock")
    if prediction_dir.is_relative_to(package) or prediction_dir.is_relative_to(source):
        raise ValueError("Predictions cannot be written inside model inputs")
    adapter_sha = hashlib.sha256(
        canonical(manifest["loader_files_sha256"]).encode("utf-8")
    ).hexdigest()
    sources = _source_hashes(source_root)
    result = {
        "schema_version": LOCK_VERSION,
        "status": "candidate_locked_before_prediction_not_release_qualified",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "candidate": {
            "key": MODEL_KEY,
            "model_id": MODEL_ID,
            "revision": revision,
            "selected_checkpoint": "checkpoint-0000368",
            "model_sha256": BEST368_SHA256,
            "package": str(package),
            "package_manifest_sha256": expected,
            "source": str(source),
            "base_repo_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "calibration_sha256": CAL_SHA256,
            "adapter_version": ADAPTER_VERSION,
            "adapter_sha256": adapter_sha,
            "loaded_parameters": LOADED_PARAMETERS,
            "parity_receipt_sha256": parity_sha256,
        },
        "comparison": comparisons,
        "comparator_roster": {
            "path": str(roster_path),
            "sha256": sha_file(roster_path),
        },
        "parity_receipt": {"path": str(parity_receipt), "sha256": parity_sha256},
        "panels": prompts,
        "prediction_paths": {
            name: str(prediction_dir / f"{MODEL_KEY}.{name}.predictions.jsonl")
            for name in PANELS
        },
        "scoring": {
            "formula": "100*sqrt(T*H)",
            "typed_items": 1600,
            "typed_answers": 2000,
            "css_items": 6547,
            "public_items_outside_aggregate": 231,
            "paired_replicates": 5000,
            "paired_seed": 20260927,
            "aggregate_gain_min_points": 3.0,
            "paired_delta_ci95_low_must_exceed_zero": True,
            "slice_regressions": "report_all; no automatic per-slice veto",
            "policy_sha256": sha_file(POLICY),
            "scorer_sources_sha256": sources,
        },
        "label_exposure": {
            "typed_final": "exposed_in_prior_4b_research",
            "css15": "exposed_in_prior_4b_research",
            "evidence_status": "same_panel_comparison_not_virgin_blind",
            "untouched_confirmation_or_explicit_disclosure_required": True,
        },
    }
    _private_write(output, result, source_root)
    return result


def prediction_seal(
    *, lock_path: Path, lock_sha256: str, output: Path, source_root: Path
) -> dict[str, Any]:
    """Audit all 8,147+231 gold-free candidate outputs before any scoring."""
    if sha_file(lock_path) != _sha(lock_sha256, "candidate lock"):
        raise ValueError("Candidate lock changed")
    lock = _object(lock_path)
    if lock.get("schema_version") != LOCK_VERSION:
        raise ValueError("Unknown 27B candidate lock")
    candidate = lock["candidate"]
    if (
        candidate.get("model_id") != MODEL_ID
        or candidate.get("model_sha256") != BEST368_SHA256
    ):
        raise ValueError("Candidate lineage differs from BEST368 lock")
    if (
        sha_file(Path(candidate["package"]) / "MODEL_MANIFEST.json")
        != candidate["package_manifest_sha256"]
    ):
        raise ValueError("Native package changed after candidate lock")
    _verify_staged_runtime(Path(candidate["package"]), Path(candidate["source"]))
    if (
        sha_file(Path(lock["parity_receipt"]["path"]))
        != lock["parity_receipt"]["sha256"]
    ):
        raise ValueError("Full-panel native parity receipt changed")
    if (
        sha_file(Path(lock["comparator_roster"]["path"]))
        != lock["comparator_roster"]["sha256"]
    ):
        raise ValueError("Comparator qualification roster changed")
    if lock["scoring"]["policy_sha256"] != sha_file(POLICY):
        raise ValueError("V3 aggregate policy changed after candidate lock")
    if lock["scoring"]["scorer_sources_sha256"] != _source_hashes(source_root):
        raise ValueError("V3 scorer or native collector changed after candidate lock")
    digests = {}
    for name, expected in PANELS.items():
        prompt = lock["panels"][name]
        rows = load_gold_free(Path(prompt["path"]))
        if len(rows) != expected or sha_file(Path(prompt["path"])) != prompt["sha256"]:
            raise ValueError(f"{name}: gold-free prompt changed")
        inputs = {row["id"]: row for row in rows}
        path = Path(lock["prediction_paths"][name])
        native_path = Path(str(path) + ".manifest.json")
        native = _object(native_path)
        if (
            native.get("model_id") != MODEL_ID
            or native.get("model_revision") != candidate["revision"]
            or native.get("model_sha256") != BEST368_SHA256
            or native.get("package_manifest_sha256")
            != candidate["package_manifest_sha256"]
            or native.get("adapter_version") != ADAPTER_VERSION
            or native.get("adapter_sha256") != candidate["adapter_sha256"]
            or native.get("calibration_sha256") != CAL_SHA256
            or native.get("base_repo_id") != BASE_ID
            or native.get("base_revision") != BASE_REVISION
            or native.get("input_sha256") != prompt["sha256"]
            or native.get("input_items") != expected
            or native.get("counts", {}).get("items") != expected
            or native.get("predictions_sha256") != sha_file(path)
        ):
            raise ValueError(f"{name}: native prediction manifest differs from lock")
        seen: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                item_id = item.get("id")
                if item_id in seen or item_id not in inputs:
                    raise ValueError(f"{name}: missing or duplicate prediction ID")
                source = inputs[item_id]
                if (
                    item.get("source_input_sha256") != input_digest(source)
                    or item.get("model_id") != MODEL_ID
                    or item.get("model_revision") != candidate["revision"]
                    or item.get("model_sha256") != BEST368_SHA256
                    or item.get("package_manifest_sha256")
                    != candidate["package_manifest_sha256"]
                    or item.get("adapter_sha256") != candidate["adapter_sha256"]
                    or item.get("calibration_sha256") != CAL_SHA256
                    or not isinstance(item.get("answers"), dict)
                    or set(item["answers"]) != set(source["questions"])
                    or set(item)
                    & {"gold", "target_probs", "teacher_probs", "provenance"}
                ):
                    raise ValueError(f"{name}: prediction differs from frozen input")
                seen.add(item_id)
        if seen != set(inputs):
            raise ValueError(f"{name}: incomplete prediction panel")
        digests[name] = {
            "prompts_sha256": prompt["sha256"],
            "predictions_sha256": sha_file(path),
            "native_manifest_sha256": sha_file(native_path),
            "items": expected,
        }
    seal = {
        "schema_version": SEAL_VERSION,
        "status": "candidate_gold_free_predictions_sealed_not_scored",
        "candidate_lock_sha256": lock_sha256,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "model_id": MODEL_ID,
        "model_revision": candidate["revision"],
        "panels": digests,
        "label_exposure": lock["label_exposure"],
    }
    _private_write(output, seal, source_root)
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    lock = actions.add_parser("lock")
    for name in (
        "package",
        "source",
        "parity_receipt",
        "roster",
        "typed_prompts",
        "css_prompts",
        "public_prompts",
        "prediction_dir",
        "source_root",
        "model_root",
        "external_root",
        "output",
    ):
        lock.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    lock.add_argument("--expected-package-sha256", required=True)
    lock.add_argument("--parity-sha256", required=True)
    seal = actions.add_parser("seal")
    for name in ("lock", "output", "source_root"):
        seal.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    seal.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    if args.command == "lock":
        result = candidate_lock(
            package=args.package,
            source=args.source,
            expected_package_sha256=args.expected_package_sha256,
            parity_receipt=args.parity_receipt,
            parity_sha256=args.parity_sha256,
            roster_path=args.roster,
            prompt_paths={
                "typed": args.typed_prompts,
                "css": args.css_prompts,
                "public": args.public_prompts,
            },
            prediction_dir=args.prediction_dir,
            source_root=args.source_root,
            model_root=args.model_root,
            external_root=args.external_root,
            output=args.output,
        )
        print(json.dumps({"status": result["status"], "sha256": sha_file(args.output)}))
    else:
        result = prediction_seal(
            lock_path=args.lock,
            lock_sha256=args.lock_sha256,
            output=args.output,
            source_root=args.source_root,
        )
        print(json.dumps({"status": result["status"], "sha256": sha_file(args.output)}))


if __name__ == "__main__":
    main()
