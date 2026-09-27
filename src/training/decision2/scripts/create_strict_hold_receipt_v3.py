"""Record the original typed Score release-gate failure without altering reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from publication.bundle_arena import _object, sha_file


def receipt(freeze_path: Path, candidate_path: Path, comparator_path: Path) -> dict:
    freeze = _object(freeze_path)
    candidate = _object(candidate_path)
    comparator = _object(comparator_path)
    if freeze.get("schema_version") != "jevarena-v3-freeze/2":
        raise ValueError("Expected the original v3 pre-key freeze")
    for report in (candidate, comparator):
        if (
            report.get("schema_version") != "typed-decision-report/2"
            or report.get("split") != "final"
            or report.get("items") != 1600
            or report.get("overall", {}).get("n") != 2000
            or report.get("gold_sha256")
            != freeze.get("panels", {}).get("typed_gold_sha256")
        ):
            raise ValueError("Typed report differs from the frozen FINAL panel")
    models = freeze.get("models", {})
    by_id = {model["model_id"]: model for model in models.values()}
    for report in (candidate, comparator):
        model = report["model"]
        frozen = by_id.get(model["id"])
        if (
            frozen is None
            or frozen["revision"] != model["revision"]
            or frozen["predictions_sha256"]["typed"] != report["predictions_sha256"]
        ):
            raise ValueError("Typed report is not from the frozen model predictions")
    new_score = candidate["by_type"]["score"]["accuracy_all"]
    old_score = comparator["by_type"]["score"]["accuracy_all"]
    if (
        type(new_score) not in (int, float)
        or type(old_score) not in (int, float)
        or not 0 <= new_score < old_score - 0.02 - 1e-12 <= 1
    ):
        raise ValueError("Original Score slice did not breach the -0.02 floor")
    return {
        "schema_version": "decision2-v3-strict-release-hold/1",
        "status": "HOLD",
        "reason": "typed_score_slice_regression",
        "prekey_freeze_sha256": sha_file(freeze_path),
        "candidate_model_id": candidate["model"]["id"],
        "comparator_model_id": comparator["model"]["id"],
        "candidate_typed_report_sha256": sha_file(candidate_path),
        "comparator_typed_report_sha256": sha_file(comparator_path),
        "candidate_score_accuracy_all": new_score,
        "comparator_score_accuracy_all": old_score,
        "predeclared_max_absolute_regression": 0.02,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--candidate-typed", type=Path, required=True)
    parser.add_argument("--comparator-typed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = receipt(args.freeze, args.candidate_typed, args.comparator_typed)
    content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)


if __name__ == "__main__":
    main()
