"""Read-only post-key diagnostic for the frozen v3 typed answer-count defect.

Run this as a separate process with the original frozen source directory. It
changes only the in-memory typed count validator; the frozen files, plan,
predictions, reports, freeze and original failed log remain untouched. The
output is a sidecar diagnostic, not a preregistered release-gate PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ERRATUM_VERSION = "jevarena-v3-postkey-answer-count-diagnostic/1"
ORIGINAL_FAILURE = "Typed overall, family and task-type counts disagree"
FAMILY_ANSWERS = {
    "constraint_competition": 400,
    "exception_stack": 400,
    "evidence_join": 800,
    "resource_ledger": 400,
}
TYPE_ANSWERS = {"choice": 800, "noul": 800, "score": 400}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corrected_typed(
    frozen: Any, report: dict[str, Any], model_id: str, revision: str
) -> tuple[float, dict[str, float]]:
    """Mirror the frozen typed check with the canonical answer count only."""
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
    overall = frozen._summary(report.get("overall"), "typed.overall")
    family, family_counts = frozen._partition(
        report.get("by_family"), set(frozen.FINAL_FAMILIES), "typed.by_family"
    )
    by_type, type_counts = frozen._partition(
        report.get("by_type"), set(frozen.TYPES), "typed.by_type"
    )
    if (
        overall[:3] != family_counts
        or overall[:3] != type_counts
        or overall[0] != 2000
        or any(
            report["by_family"][name]["n"] != count
            for name, count in FAMILY_ANSWERS.items()
        )
        or any(
            report["by_type"][name]["n"] != count
            for name, count in TYPE_ANSWERS.items()
        )
    ):
        raise ValueError("Typed overall, family and task-type counts disagree")
    return (
        frozen._close(
            report.get("macro_family_accuracy"),
            statistics.mean(family.values()),
            "typed.macro_family_accuracy",
        ),
        by_type,
    )


def diagnose(source_root: Path, roster: Path, failed_log: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    source = source_root / "jev_arena/arena_v3.py"
    if not source.is_file() or source.is_symlink():
        raise ValueError("Original frozen v3 ranker source is missing or linked")
    if not failed_log.is_file() or failed_log.is_symlink():
        raise ValueError("Original failed rank log is missing or linked")
    sys.path.insert(0, str(source_root))
    frozen = importlib.import_module("jev_arena.arena_v3")
    if Path(frozen.__file__).resolve() != source:
        raise ValueError("Imported ranker is not the supplied frozen source")
    try:
        frozen.rank(roster)
    except ValueError as error:
        if str(error) != ORIGINAL_FAILURE:
            raise ValueError("Frozen ranker failed for a different reason") from error
    else:
        raise ValueError("Frozen ranker did not reproduce the count defect")
    original = frozen._typed
    try:
        frozen._typed = lambda report, model_id, revision: corrected_typed(
            frozen, report, model_id, revision
        )
        ranked = frozen.rank(roster)
    finally:
        frozen._typed = original
    if ranked.get("schema_version") != "jevarena-ranking/3":
        raise ValueError("Unexpected frozen ranking schema")
    return {
        "schema_version": ERRATUM_VERSION,
        "status": "postkey_diagnostic_not_preregistered_release_gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_frozen_ranker_sha256": sha(source),
        "original_failed_rank_log_sha256": sha(failed_log),
        "erratum_script_sha256": sha(Path(__file__)),
        "roster_sha256": sha(roster),
        "prekey_freeze_sha256": ranked["freeze_sha256"],
        "correction": "Typed FINAL has 1,600 input items and 2,000 scored answers; only the frozen ranker's typed item/answer count and partition-shape validator is replaced in memory.",
        "ranked_result": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--original-failed-rank-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        args.frozen_source_root,
        args.roster,
        args.original_failed_rank_log,
    )
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
