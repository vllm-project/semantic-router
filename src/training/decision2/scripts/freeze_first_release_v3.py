"""Seal a JevArena v3 pre-key receipt from fully audited gold-free outputs.

Only typed/CSS gold SHA-256 strings are accepted; this module has no gold path
argument and never opens either label file. The resulting receipt records a
declared event chronology, not independently trusted timestamp evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_arena.arena_v3 import (
    FREEZE_VERSION,
    SCORER_SOURCE_PATHS,
    _digest,
    _freeze,
    _prekey_evidence,
)
from scripts.plan_first_release_v3 import (
    PLAN_VERSION,
    audit_prekey_predictions,
    pair_digest,
    sha_file,
)


def _private_input(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name}: expected a regular absolute private file")
    if sha_file(path) != _digest(expected_sha256, f"{name}.sha256"):
        raise ValueError(f"{name}: saved digest changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return value


def _private_output(path: Path, source_root: Path) -> None:
    if not path.is_absolute() or path.parent.is_symlink():
        raise ValueError("Output must be under a private, absolute directory")
    directory = path.parent.resolve(strict=True)
    if directory.is_relative_to(source_root.resolve(strict=True)):
        raise ValueError("Pre-key receipt must not be written into the source tree")
    mode = directory.stat().st_mode
    if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) & 0o077:
        raise ValueError("Pre-key receipt directory must be private (0700)")
    if directory.stat().st_uid != os.getuid():
        raise ValueError("Pre-key receipt directory has another owner")
    if path.exists() or path.is_symlink():
        raise FileExistsError("Pre-key receipt already exists")


def create_freeze(
    *,
    plan_path: Path,
    plan_sha256: str,
    audit_path: Path,
    audit_sha256: str,
    chronology_path: Path,
    chronology_sha256: str,
    typed_gold_sha256: str,
    css_gold_sha256: str,
    output: Path,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Reaudit every prediction before one exclusive, private receipt write."""
    plan = _private_input(plan_path, plan_sha256, "plan")
    audit = _private_input(audit_path, audit_sha256, "audit")
    _private_input(chronology_path, chronology_sha256, "chronology")
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError("Unknown first-release v3 plan")
    _digest(typed_gold_sha256, "typed_gold_sha256")
    _digest(css_gold_sha256, "css_gold_sha256")
    source_root = Path(plan["source_root"])
    _private_output(output, source_root)

    # This reopens only gold-free prompts, native predictions, package receipts,
    # and source files. It enforces 1600 + 6547 + 231 rows for every model.
    observed = audit_prekey_predictions(copy.deepcopy(plan))
    if observed != audit:
        raise ValueError("Saved audit differs from the current full prediction audit")
    roster = plan.get("model_roster")
    if not isinstance(roster, list) or len(roster) < 3:
        raise ValueError("Plan lacks candidate, Decision 1.0, or open control")
    if not {"decision2", "decision1", "open"}.issubset(
        {row.get("group") for row in roster if isinstance(row, dict)}
    ):
        raise ValueError("Plan lacks candidate, Decision 1.0, or open control")
    keys = {row["key"] for row in roster}
    if len(keys) != len(roster) or set(audit.get("models", {})) != keys:
        raise ValueError("Saved audit does not cover every planned model")
    if pair_digest(plan.get("comparison_pairs", [])) != plan.get(
        "comparison_pairs_sha256"
    ):
        raise ValueError("Predeclared comparison mapping changed")

    frozen_at = now or datetime.now(timezone.utc)
    if frozen_at.tzinfo is None or frozen_at.utcoffset() != timezone.utc.utcoffset(
        None
    ):
        raise ValueError("Freeze clock must be UTC")
    frozen_time = frozen_at.isoformat(timespec="microseconds")
    models = {
        row["key"]: {
            name: row[name]
            for name in (
                "model_id",
                "revision",
                "native_model_sha256",
                "adapter_sha256",
                "calibration_sha256",
            )
        }
        | {"predictions_sha256": audit["models"][row["key"]]}
        for row in roster
    }
    freeze = {
        "schema_version": FREEZE_VERSION,
        "status": "prekey_frozen",
        "plan": {"path": str(plan_path), "sha256": plan_sha256},
        "prediction_audit": {"path": str(audit_path), "sha256": audit_sha256},
        "chronology": {"path": str(chronology_path), "sha256": chronology_sha256},
        "comparison_pairs": plan["comparison_pairs"],
        "comparison_pairs_sha256": plan["comparison_pairs_sha256"],
        "prekey_frozen_at_utc": frozen_time,
        "prompt_sha256": audit["prompt_sha256"],
        "raw_prediction_hashes_sha256": audit["raw_hashes_sha256"],
        "score_sources_sha256": {
            name: sha_file(path) for name, path in SCORER_SOURCE_PATHS.items()
        },
        "protocol_sha256": plan["gate_document_sha256"],
        "candidate_lock_sha256": plan["candidate_freeze_sha256"],
        "formula": "100*sqrt(T*H)",
        "paired_bootstrap": {"replicates": 5000, "seed": 20260927},
        "panels": {
            "typed_gold_sha256": typed_gold_sha256,
            "css_gold_sha256": css_gold_sha256,
        },
        "models": models,
    }
    _prekey_evidence(freeze, output, keys)
    encoded = (json.dumps(freeze, sort_keys=True, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(output, flags, 0o600), "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    receipt_sha256 = sha_file(output)
    _freeze(output, {"freeze_sha256": receipt_sha256}, keys)
    return receipt_sha256, frozen_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prediction-audit", type=Path, required=True)
    parser.add_argument("--prediction-audit-sha256", required=True)
    parser.add_argument("--chronology", type=Path, required=True)
    parser.add_argument("--chronology-sha256", required=True)
    parser.add_argument("--typed-gold-sha256", required=True)
    parser.add_argument("--css-gold-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        digest, frozen_at = create_freeze(
            plan_path=args.plan,
            plan_sha256=args.plan_sha256,
            audit_path=args.prediction_audit,
            audit_sha256=args.prediction_audit_sha256,
            chronology_path=args.chronology,
            chronology_sha256=args.chronology_sha256,
            typed_gold_sha256=args.typed_gold_sha256,
            css_gold_sha256=args.css_gold_sha256,
            output=args.output,
        )
    except (OSError, KeyError, TypeError, ValueError):
        # Do not echo private paths, model IDs, prediction contents, or labels.
        raise SystemExit("pre-key freeze validation failed") from None
    print(
        json.dumps({"status": "prekey_frozen", "sha256": digest, "at_utc": frozen_at})
    )


if __name__ == "__main__":
    main()
