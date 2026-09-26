"""Audit authored panel template reuse and prepare a blind 20-item QA sample.

This reads only the new authored panel's own source specs, prompts and targets.
It never opens the preexisting FINAL/CSS gold or model predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from .sealed_authored import (
    _load_panel,
    _read_jsonl,
    compact,
    sha_file,
    solve_rendered,
    solve_spec,
    DOMAINS,
)

AUDIT_VERSION = "jevarena-authored-template-diversity/1"


def semantic_signature(prompt: dict[str, Any], spec: dict[str, Any]) -> str:
    """Remove IDs and domain words, retaining the operative policy/question."""
    policy = prompt["state"].split("\nPolicy: ", 1)[1].split("\nLedger:\n", 1)[0]
    question = prompt["questions"]["decision"]["instructions"]
    domain = DOMAINS[spec["domain"]]
    for index, check in sorted(
        enumerate(domain["checks"]), key=lambda pair: -len(pair[1])
    ):
        policy = policy.replace(check, f"<CHECK{index}>")
    for index, action in sorted(
        enumerate(spec["action_keys"]), key=lambda pair: -len(pair[1])
    ):
        policy = policy.replace(action, f"<ACTION{index}>")
    question = question.replace(spec["target_id"], "<TARGET>")
    return hashlib.sha256(
        compact([spec["type"], policy, question]).encode()
    ).hexdigest()


def choose_sample(targets: list[dict[str, Any]]) -> list[int]:
    """Deterministic family/domain coverage, then two globally hashed cases."""
    chosen: set[int] = set()

    def rank(index: int) -> str:
        return hashlib.sha256(targets[index]["id"].encode()).hexdigest()

    for family in sorted({row["family"] for row in targets}):
        eligible = [
            index for index, row in enumerate(targets) if row["family"] == family
        ]
        chosen.add(min(eligible, key=rank))
    for domain in DOMAINS:
        eligible = [
            index
            for index, row in enumerate(targets)
            if row["domain"] == domain and index not in chosen
        ]
        chosen.add(min(eligible, key=rank))
    for index in sorted((i for i in range(len(targets)) if i not in chosen), key=rank):
        if len(chosen) == 20:
            break
        chosen.add(index)
    if len(chosen) != 20:
        raise ValueError("Not enough authored cases for 20-item QA sample")
    return sorted(chosen)


def audit(panel: Path, report_path: Path, sample_path: Path) -> dict[str, Any]:
    if report_path.exists() or sample_path.exists():
        raise FileExistsError("Template audit refuses to overwrite results")
    manifest, automated, prompts, targets = _load_panel(panel)
    specs = _read_jsonl(panel / "source_specs.jsonl")
    if len(specs) != len(prompts):
        raise ValueError("Authored source-spec row count differs")
    signatures: dict[tuple[str, str, bool | None], set[str]] = defaultdict(set)
    core = Counter()
    styles = Counter()
    operations = Counter()
    challenge_checks = Counter()
    target_positions = []
    sample_indices = choose_sample(targets)
    for prompt, target, spec in zip(prompts, targets, specs):
        if not prompt["id"] == target["id"] == spec["id"]:
            raise ValueError("Authored source/prompt/target ID mismatch")
        if solve_spec(spec) != solve_rendered(prompt, target):
            raise ValueError("Template audit found an ambiguous rendered answer")
        kind, challenge = spec["type"], spec["challenge"]
        polarity = spec["ask_sufficient"] if kind == "noul" else None
        key = (kind, challenge, polarity)
        core[key] += 1
        styles[spec["style"]] += 1
        operations[kind] += 1
        signatures[key].add(semantic_signature(prompt, spec))
        rows = spec["rows"]
        operative = [
            index
            for index, row in enumerate(rows)
            if row["role"] == "FINAL" and row["id"] == spec["target_id"]
        ]
        if len(operative) != 1:
            raise ValueError("Template audit found nonunique operative FINAL row")
        if challenge == "long_context":
            if len(rows) != 193 or len(prompt["state"].split()) < 2800:
                raise ValueError("Long-context challenge lacks preregistered length")
            target_positions.append(operative[0] / (len(rows) - 1))
        elif challenge == "near_distractor":
            target_hex = int(spec["target_id"][-10:], 16)
            expected = {(target_hex + offset) % (1 << 40) for offset in range(1, 6)}
            actual = {
                int(row["id"][-10:], 16)
                for row in rows
                if row["id"] != spec["target_id"]
            }
            if not expected <= actual:
                raise ValueError("Near-distractor challenge lacks adjacent identifiers")
        elif challenge == "insufficient_evidence":
            if "unreported" not in spec["flags"] or not any(
                row["role"] == "UNVERIFIED"
                and row["id"] == spec["target_id"]
                and row["flags"] == ["yes"] * 4
                for row in rows
            ):
                raise ValueError("Insufficient-evidence challenge is not operative")
        elif challenge == "rule_precedence":
            if not any(
                row["role"] == "DRAFT"
                and row["id"] == spec["target_id"]
                and row["flags"] != spec["flags"]
                for row in rows
            ):
                raise ValueError("Rule-precedence challenge lacks conflicting draft")
        else:
            raise ValueError("Unexpected authored challenge")
        challenge_checks[challenge] += 1
    if (
        len(core) != 16
        or len(signatures) != 16
        or any(len(v) != 1 for v in signatures.values())
    ):
        raise ValueError("Template inventory changed or domain substitution differs")
    if set(styles) != set(range(4)) or set(operations) != {"choice", "noul", "score"}:
        raise ValueError("Template style or semantic operation inventory changed")
    sample = [
        {
            "id": targets[i]["id"],
            "cell": "/".join(
                (targets[i]["task_type"], targets[i]["challenge"], targets[i]["domain"])
            ),
            "prompt": prompts[i],
            "gold": targets[i]["gold"]["decision"],
        }
        for i in sample_indices
    ]
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("x", encoding="utf-8") as stream:
        for row in sample:
            stream.write(compact(row) + "\n")
    sample_path.chmod(0o600)
    report = {
        "audit_version": AUDIT_VERSION,
        "phase": manifest["phase"],
        "items": len(prompts),
        "manifest_sha256": sha_file(panel / "manifest.json"),
        "automated_audit_sha256": manifest["automated_audit_sha256"],
        "automated_status": automated["automated_status"],
        "semantic_operations": dict(sorted(operations.items())),
        "logical_question_templates": len(core),
        "template_families_type_by_challenge": 12,
        "domain_lexical_substitutions": len(DOMAINS),
        "surface_header_styles": dict(sorted(styles.items())),
        "normalized_policy_question_signatures": len(
            {next(iter(v)) for v in signatures.values()}
        ),
        "cross_domain_policy_equivalent_for_each_core_template": all(
            len(v) == 1 for v in signatures.values()
        ),
        "challenge_invariant_checks": dict(sorted(challenge_checks.items())),
        "long_context_target_position_fraction": {
            "min": min(target_positions),
            "median": statistics.median(target_positions),
            "max": max(target_positions),
        },
        "sample20_ids": [row["id"] for row in sample],
        "sample20_packet_sha256": sha_file(sample_path),
        "quality_interpretation": (
            "Unique record instantiations share only 16 logical-question templates "
            "and three semantic operations. Domain differences are lexical substitutions; "
            "do not interpret 1296 groups as 1296 independent scenario designs. "
            "Human editorial review remains required."
        ),
        "template_quality_gate": {
            "status": "blocked",
            "reason": "limited_template_diversity_and_human_review_pending",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample20-private", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.panel_dir, args.report, args.sample20_private)
    print(
        compact(
            {
                "items": report["items"],
                "logical_question_templates": report["logical_question_templates"],
                "template_quality_gate": report["template_quality_gate"],
            }
        )
    )


if __name__ == "__main__":
    main()
