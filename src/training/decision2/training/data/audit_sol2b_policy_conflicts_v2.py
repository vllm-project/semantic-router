"""One-shot v2 shortcut, token-capacity and preliminary overlap audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from training.data.audit_sol2b_policy_conflicts import (
    CONTROL_TRAIN_SHA256,
    file_sha256,
    shortcut_probe,
    token_feasibility,
    verify_blind_packet,
)
from training.data.build_sol2b_policy_conflicts_v2 import (
    DIAGNOSTIC_SEED,
    build_all,
)
from training.model.data import canonical, load_partition


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _text(value: Any) -> str:
    return value if isinstance(value, str) else canonical(value)


def _normalized_input(row: dict[str, Any]) -> str:
    options = " ".join(
        f"{_text(option['key'])} {_text(option['description'])}"
        for option in row["options"]
    )
    return _normalized(
        " ".join((_text(row["state"]), _text(row["instructions"]), options))
    )


def preliminary_overlap(
    candidate: list[dict[str, Any]], comparisons: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, int]]:
    """Count exact/normalized reuse; near and semantic review remain separate."""
    result = {}
    candidate_ids = {row["id"] for row in candidate}
    candidate_input = {row["input_sha256"] for row in candidate}
    candidate_normalized = {_normalized_input(row) for row in candidate}
    candidate_states = {_normalized(_text(row["state"])) for row in candidate}
    candidate_facts = {row["audit_metadata"]["fact_digest"] for row in candidate}
    for name, rows in comparisons.items():
        result[name] = {
            "rows": len(rows),
            "shared_row_id": len(candidate_ids & {row["id"] for row in rows}),
            "shared_input_sha256": len(
                candidate_input & {row["input_sha256"] for row in rows}
            ),
            "shared_normalized_input": len(
                candidate_normalized & {_normalized_input(row) for row in rows}
            ),
            "shared_normalized_state": len(
                candidate_states & {_normalized(_text(row["state"])) for row in rows}
            ),
            "shared_fact_digest": len(
                candidate_facts
                & {row.get("audit_metadata", {}).get("fact_digest") for row in rows}
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--v1-candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()

    if file_sha256(args.control) != CONTROL_TRAIN_SHA256:
        raise ValueError("Frozen control TRAIN SHA changed")
    candidate = load_partition(args.candidate, "train")
    control = load_partition(args.control, "train")
    blind = _jsonl(args.blind)
    diagnostic, _ = build_all(groups_per_family=32, seed=DIAGNOSTIC_SEED)
    if {row["input_sha256"] for row in candidate} & {
        row["input_sha256"] for row in diagnostic
    }:
        raise ValueError("Shortcut diagnostic shares a training input")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    report = {
        "schema_version": "decision2-sol2b-policy-v2-cpu-audit/1",
        "candidate_sha256": file_sha256(args.candidate),
        "blind_sha256": file_sha256(args.blind),
        "blind": verify_blind_packet(blind),
        "candidate_groups": len({row["group_id"] for row in candidate}),
        "diagnostic_groups": len({row["group_id"] for row in diagnostic}),
        "shortcut": shortcut_probe(diagnostic),
        "tokens": token_feasibility(candidate, control, tokenizer),
        "overlap_preliminary": preliminary_overlap(
            candidate,
            {
                "v1_candidate": load_partition(args.v1_candidate, "train"),
                "control_train": control,
                "control_select": _jsonl(args.select),
                "control_cal": _jsonl(args.cal),
            },
        ),
        "near_and_semantic_overlap_reviewed": False,
    }
    print(canonical(report))


if __name__ == "__main__":
    main()
