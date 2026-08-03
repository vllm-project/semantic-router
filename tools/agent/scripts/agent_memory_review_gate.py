#!/usr/bin/env python3
"""Evaluate hard gates for memory-assisted PR review output."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MATCH_STATUS_PATTERN = re.compile(
    r"(?im)^\s*(?:-\s*)?Review brief matches diff\s*:\s*(?P<value>[A-Za-z_-]+)\s*$"
)
HARD_GATE_PATTERN = re.compile(
    r"(?im)^\s*(?:-\s*)?Hard gate\s*:\s*(?P<value>[A-Za-z_-]+)\s*$"
)
REASON_PATTERN = re.compile(r"(?im)^\s*(?:-\s*)?Reason\s*:\s*(?P<value>.+?)\s*$")

PASS_VALUES = {"pass", "passed", "yes", "true", "aligned", "match", "matches"}
FAIL_VALUES = {"fail", "failed", "no", "false", "mismatch", "conflict"}
NOT_APPLICABLE_VALUES = {"not-applicable", "not_applicable", "na", "n/a"}


@dataclass(frozen=True)
class ReviewGateResult:
    gate_passed: bool
    gate_reason: str | None
    review_brief_match: str
    comment_body: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_value(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match:
        return match.group("value").strip()
    return None


def classifier_gate_reason(classifier: dict[str, Any]) -> str | None:
    if classifier.get("gate_passed") is False:
        return str(classifier.get("gate_reason") or "Review brief hard gate failed")
    if classifier.get("memory_invalid"):
        return str(
            classifier.get("invalid_reason") or "Review brief reference is invalid"
        )
    if classifier.get("memory_required") and not classifier.get("memory_present"):
        return "Review brief is required for this PR size but no valid brief was found"
    return None


def parse_review_response(response: str) -> tuple[str, str | None]:
    match_value = normalize_value(first_match(MATCH_STATUS_PATTERN, response))
    hard_gate_value = normalize_value(first_match(HARD_GATE_PATTERN, response))
    reason = first_match(REASON_PATTERN, response)

    if match_value in FAIL_VALUES or hard_gate_value in FAIL_VALUES:
        return "mismatch", reason or "Review brief does not match this PR diff"
    if match_value in PASS_VALUES and (
        hard_gate_value is None or hard_gate_value in PASS_VALUES
    ):
        return "match", None
    if match_value in NOT_APPLICABLE_VALUES:
        return "not-applicable", None
    return "unknown", "Memory-assisted review did not emit a parseable gate verdict"


def build_comment(gate_passed: bool, reason: str | None, status: str) -> str:
    lines = ["## Agent Memory Hard Gate", ""]
    if gate_passed:
        lines.append("- Status: PASS")
    else:
        lines.append("- Status: FAIL")
        lines.append(f"- Problem: {reason}")
    lines.append(f"- Review brief match: {status}")
    if status == "unknown" and gate_passed:
        lines.append(
            "- Note: AI review unavailable; classifier hard gate already passed."
        )
    return "\n".join(lines)


def evaluate_gate(
    *,
    classifier: dict[str, Any],
    review_response: str | None,
    classifier_only: bool = False,
    advisory_on_review_unavailable: bool = False,
) -> ReviewGateResult:
    reason = classifier_gate_reason(classifier)
    if reason is not None:
        return ReviewGateResult(
            gate_passed=False,
            gate_reason=reason,
            review_brief_match="invalid-or-missing",
            comment_body=build_comment(False, reason, "invalid-or-missing"),
        )

    if not classifier.get("memory_present"):
        return ReviewGateResult(
            gate_passed=True,
            gate_reason=None,
            review_brief_match="not-applicable",
            comment_body=build_comment(True, None, "not-applicable"),
        )

    if classifier_only:
        return ReviewGateResult(
            gate_passed=True,
            gate_reason=None,
            review_brief_match="not-evaluated",
            comment_body=build_comment(True, None, "not-evaluated"),
        )

    return evaluate_review_response_gate(
        review_response=review_response,
        advisory_on_review_unavailable=advisory_on_review_unavailable,
    )


def evaluate_review_response_gate(
    *,
    review_response: str | None,
    advisory_on_review_unavailable: bool,
) -> ReviewGateResult:
    if not review_response:
        if advisory_on_review_unavailable:
            # Rollout mode: the deterministic classifier has already passed, so
            # missing model output is reported as unknown/advisory instead of
            # failing merge on GitHub Models availability.
            return ReviewGateResult(
                gate_passed=True,
                gate_reason=None,
                review_brief_match="unknown",
                comment_body=build_comment(True, None, "unknown"),
            )
        # Default mode remains fail-closed for callers that require the AI
        # brief/diff verdict to be present and parseable.
        reason = (
            "Memory-assisted review did not produce a response, so brief/diff "
            "alignment could not be verified"
        )
        return ReviewGateResult(
            gate_passed=False,
            gate_reason=reason,
            review_brief_match="unknown",
            comment_body=build_comment(False, reason, "unknown"),
        )

    status, parsed_reason = parse_review_response(review_response)
    if status == "mismatch":
        reason = parsed_reason or "Review brief does not match this PR diff"
        return ReviewGateResult(
            gate_passed=False,
            gate_reason=reason,
            review_brief_match=status,
            comment_body=build_comment(False, reason, status),
        )
    if status == "not-applicable":
        reason = (
            "Memory-assisted review marked brief/diff consistency as "
            "not-applicable even though a review brief is present"
        )
        return ReviewGateResult(
            gate_passed=False,
            gate_reason=reason,
            review_brief_match="unknown",
            comment_body=build_comment(False, reason, "unknown"),
        )
    if status == "unknown":
        reason = parsed_reason or "Memory-assisted review gate verdict is unknown"
        return ReviewGateResult(
            gate_passed=False,
            gate_reason=reason,
            review_brief_match=status,
            comment_body=build_comment(False, reason, status),
        )
    return ReviewGateResult(
        gate_passed=True,
        gate_reason=None,
        review_brief_match=status,
        comment_body=build_comment(True, None, status),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True, type=Path)
    parser.add_argument("--review-response", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Only enforce classifier hard-gate output; skip AI brief/diff verdict checks.",
    )
    parser.add_argument(
        "--advisory-on-review-unavailable",
        action="store_true",
        help=(
            "Pass when classifier hard gate passed but AI review output is unavailable; "
            "keeps review-status comments advisory during rollout."
        ),
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    review_response = None
    if args.review_response and args.review_response.is_file():
        review_response = args.review_response.read_text(
            encoding="utf-8", errors="replace"
        )
    result = evaluate_gate(
        classifier=load_json(args.classifier),
        review_response=review_response,
        classifier_only=args.classifier_only,
        advisory_on_review_unavailable=args.advisory_on_review_unavailable,
    )
    args.output.write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )

    if args.fail_on_gate and not result.gate_passed:
        print(result.gate_reason)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
