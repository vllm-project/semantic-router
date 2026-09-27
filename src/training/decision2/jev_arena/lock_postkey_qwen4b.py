"""Gold-free, prospective post-key lock for the official-source Qwen 4B arm.

This module never accepts a gold path. Earlier Nox/Kev predictions may be
reused only when every item, question, prompt digest, model revision, and byte
hash matches their archived pre-key freeze.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from inference.run import file_digest
from training.data.audit_release4b_overlap import (
    EXPECTED_N,
    EXPECTED_SHA,
    audit_panel,
    read_jsonl,
)
from training.model.data import canonical
from training.model.infer import checkpoint_fingerprint

from .seal_postkey_qwen06b import (
    PANELS,
    _control_rows,
    _exclusive_json,
    _object,
    _prompts,
    _source,
)

SCHEMA = "decision2-official-qwen4b-v3-postkey-lock/1"
ROSTER_SCHEMA = "decision2-official-qwen4b-v3-postkey-roster/1"


def overlap(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "train": args.train,
        "typed_final": args.typed_prompts,
        "css15": args.css_prompts,
        "public231": args.public_prompts,
    }
    for name, path in paths.items():
        if file_digest(path) != EXPECTED_SHA[name]:
            raise ValueError(f"Frozen {name} hash differs")
    train = read_jsonl(args.train, prompt=False)
    panels = {
        name: read_jsonl(paths[name], prompt=True)
        for name in ("typed_final", "css15", "public231")
    }
    for name, rows in {"train": train, **panels}.items():
        if len(rows) != EXPECTED_N[name]:
            raise ValueError(f"{name} row count differs")
    audits = {name: audit_panel(train, rows) for name, rows in panels.items()}
    result = {
        "schema": "decision2-official-qwen4b-goldfree-overlap/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_sha256": {name: file_digest(path) for name, path in paths.items()},
        "rows": {
            "train": len(train),
            **{name: len(rows) for name, rows in panels.items()},
        },
        "panels": audits,
        "limits": "State-only approximate search cannot prove semantic disjointness or upstream pretraining isolation.",
        "gold_or_scores_read": False,
    }
    _exclusive_json(args.output, result)
    return result


def _model(roster: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    selected = roster["candidate"]
    best, complete, cal = (
        _object(args.best),
        _object(args.complete),
        _object(args.calibration),
    )
    if (
        file_digest(args.best) != selected["best_sha256"]
        or file_digest(args.complete) != selected["complete_sha256"]
        or best.get("checkpoint") != f"checkpoint-{selected['best_step']:07d}"
        or complete.get("best") != best["checkpoint"]
        or complete.get("status") != "complete"
        or file_digest(args.calibration) != selected["calibration_sha256"]
        or cal.get("selected_checkpoint") != best["checkpoint"]
        or cal.get("fit_split") != "cal"
        or cal.get("best_sha256") != selected["best_sha256"]
        or cal.get("complete_sha256") != selected["complete_sha256"]
        or cal.get("cal_sha256") != selected["cal_sha256"]
    ):
        raise ValueError("BEST, COMPLETE, or disjoint CAL binding differs")
    run = _object(args.training_provenance)
    if (
        file_digest(args.training_provenance) != selected["training_provenance_sha256"]
        or run.get("contract", {}).get("data_sha256", {}).get("train")
        != selected["train_sha256"]
        or run.get("contract", {}).get("data_sha256", {}).get("cal")
        != selected["cal_sha256"]
        or run.get("contract", {}).get("base_revision") != selected["base_revision"]
        or run.get("contract", {}).get("init_kind") != "base"
    ):
        raise ValueError("Training source/data lineage differs")
    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    if (
        identity["model_sha256"] != selected["model_sha256"]
        or cal.get("model_sha256") != selected["model_sha256"]
    ):
        raise ValueError("Selected checkpoint/CAL model fingerprint differs")
    for relative, expected in identity["files_sha256"].items():
        origin = (
            args.checkpoint if relative.startswith("checkpoint/") else args.source_path
        )
        suffix = relative.split("/", 1)[1]
        if file_digest(origin / suffix) != expected:
            raise ValueError("Source/checkpoint file changed after fingerprint")
    if (
        hashlib.sha256(canonical(identity["files_sha256"]).encode()).hexdigest()
        != selected["model_sha256"]
    ):
        raise ValueError("LoRA source and checkpoint closure differs")
    parameters = _object(args.parameter_receipt)
    if (
        file_digest(args.parameter_receipt) != selected["parameter_receipt_sha256"]
        or parameters.get("model_sha256") != selected["model_sha256"]
        or parameters.get("loaded_parameters") != selected["loaded_parameters"]
    ):
        raise ValueError("Measured loaded parameter count differs")
    repeat = _object(args.repeat_receipt)
    if (
        file_digest(args.repeat_receipt) != selected["repeat_receipt_sha256"]
        or repeat.get("passed") is not True
        or repeat.get("model_sha256") != selected["model_sha256"]
        or repeat.get("calibration_sha256") != selected["calibration_sha256"]
        or repeat.get("rows") != 32
    ):
        raise ValueError("CAL-bound repeated native inference failed")
    return {
        "model_sha256": identity["model_sha256"],
        "model_files_sha256": identity["files_sha256"],
        "loaded_parameters": parameters["loaded_parameters"],
        "calibration_sha256": file_digest(args.calibration),
        "repeat_receipt_sha256": file_digest(args.repeat_receipt),
    }


def lock(args: argparse.Namespace) -> dict[str, Any]:
    roster = _object(args.roster)
    if (
        roster.get("schema") != ROSTER_SCHEMA
        or roster.get("not_a_never_unsealed_blind_test") is not True
        or roster.get("paired_bootstrap") != {"replicates": 5000, "seed": 20260927}
        or roster.get("score_rule")
        != "100 * sqrt(typed_FINAL_four_family_macro_accuracy * CSS15_task_median_macro_F1)"
        or roster.get("invalid_policy")
        != "missing, invalid and over-budget answers fail at full denominator"
        or roster.get("release_gate")
        != "candidate JevArena v3 main score >= same-panel own Nox 1.0 score + 3.0 points; disclose all regressions"
    ):
        raise ValueError("Wrong prospective post-key protocol")
    _source(roster, args.source_root)
    panel_paths = {name: getattr(args, f"{name}_prompts") for name in PANELS}
    prompts = _prompts(roster, panel_paths)
    if (
        file_digest(args.public_manifest)
        != roster["panel"]["public"]["manifest_sha256"]
    ):
        raise ValueError("Public-231 manifest changed")
    audit = _object(args.overlap_audit)
    if (
        file_digest(args.overlap_audit) != roster["overlap_audit_sha256"]
        or audit.get("schema") != "decision2-official-qwen4b-goldfree-overlap/1"
        or audit.get("gold_or_scores_read") is not False
    ):
        raise ValueError("TRAIN/eval overlap audit missing or changed")
    for panel, path in (
        ("typed_final", args.typed_prompts),
        ("css15", args.css_prompts),
        ("public231", args.public_prompts),
    ):
        if audit["input_sha256"][panel] != file_digest(path):
            raise ValueError("Overlap audit used other prompts")
        if any(audit["panels"][panel]["counts"].values()):
            raise ValueError("Observed TRAIN/eval overlap")
    if audit["input_sha256"]["train"] != roster["candidate"]["train_sha256"]:
        raise ValueError("Overlap audit used other TRAIN")
    model = _model(roster, args)
    freeze = _object(args.prior_freeze)
    prekey = _object(args.prior_prediction_audit)
    if (
        file_digest(args.prior_freeze) != roster["prior_freeze_sha256"]
        or file_digest(args.prior_prediction_audit)
        != freeze["prediction_audit"]["sha256"]
        or prekey.get("status") != "gold_free_prekey_predictions_verified"
        or prekey.get("prompt_sha256")
        != {name: file_digest(path) for name, path in panel_paths.items()}
        or freeze.get("formula") != "100*sqrt(T*H)"
    ):
        raise ValueError("Archived comparator freeze/prediction audit differs")
    controls = {}
    for key in ("nox", "kev"):
        expected = roster["comparators"][key]
        archived = freeze["models"][key]
        if (
            expected["model_id"] != archived["model_id"]
            or expected["weight_revision"] != archived["revision"]
            or expected["prediction_sha256"] != archived["predictions_sha256"]
            or expected["adapter_sha256"] != archived["adapter_sha256"]
            or expected["calibration_sha256"] != archived["calibration_sha256"]
        ):
            raise ValueError(f"{key} identity differs from archived freeze")
        controls[key] = {}
        for panel in PANELS:
            path = args.prior_predictions / f"{key}.{panel}.predictions.jsonl"
            observed = _control_rows(path, prompts[panel], expected)
            if (
                observed["sha256"] != expected["prediction_sha256"][panel]
                or observed["sha256"] != prekey["models"][key][panel]
            ):
                raise ValueError(f"{key}/{panel} differs from archived seal")
            controls[key][panel] = observed
    result = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "post_key_same_panel": True,
        "not_a_never_unsealed_blind_test": True,
        "roster_sha256": file_digest(args.roster),
        "code_sha256": file_digest(Path(__file__)),
        "candidate": model,
        "overlap_audit_sha256": file_digest(args.overlap_audit),
        "prompts_sha256": {
            name: file_digest(path) for name, path in panel_paths.items()
        },
        "controls": controls,
        "scoring_sources_sha256": roster["scoring_sources_sha256"],
        "claim": "Candidate locked before its formal/public predictions; project labels previously accessed in other experiments.",
    }
    _exclusive_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    audit = sub.add_parser("overlap")
    for name in ("train", "typed_prompts", "css_prompts", "public_prompts", "output"):
        audit.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    locked = sub.add_parser("lock")
    for name in (
        "roster",
        "source_root",
        "typed_prompts",
        "css_prompts",
        "public_prompts",
        "public_manifest",
        "overlap_audit",
        "checkpoint",
        "source_path",
        "training_provenance",
        "calibration",
        "best",
        "complete",
        "parameter_receipt",
        "repeat_receipt",
        "prior_freeze",
        "prior_prediction_audit",
        "prior_predictions",
        "output",
    ):
        locked.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = overlap(args) if args.phase == "overlap" else lock(args)
    print(json.dumps({"schema": result["schema"], "sha256": file_digest(args.output)}))


if __name__ == "__main__":
    main()
