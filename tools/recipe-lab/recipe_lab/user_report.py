"""Render selected Recipe Lab receipts without reading configs, traces or task text."""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Finding:
    title: str
    purpose: str
    outcome: str
    details: list[str]
    next_step: str


def _object(value):
    if not isinstance(value, dict):
        raise ValueError("expected an evidence object")
    return value


def _items(value):
    if not isinstance(value, list):
        raise ValueError("expected an evidence list")
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative evidence count")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite measurement")
    return value


def _fmt(value):
    return "unavailable" if value is None else f"{_number(value):.6g}"


def _choice(value, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError("unrecognized receipt status")
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("expected a SHA-256 in the receipt")
    return f"`{value}`"


def _hashes(row, fields):
    return [
        f"{label} SHA-256 {_hash(row[key])}."
        for key, label in fields
        if row.get(key) is not None
    ]


_KEY = re.compile(
    r"(?:\b(?:sk-|hf_|ghp_|github_pat_|AKIA)[A-Za-z0-9_-]{8,}"
    r"|\bBearer\s+\S+|[\w.+-]*(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


def _label(value):
    # Printable identifiers only. Never interpolate arbitrary paths, URLs or errors.
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[a-zA-Z0-9_-][a-zA-Z0-9._-]*(?:[:/][a-zA-Z0-9_-][a-zA-Z0-9._-]*)*", value
        )
        or len(value) > 128
        or _KEY.search(value)
        or re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", value)
    ):
        return "(identifier omitted)"
    return f"`{value}`"


def _bench(row):
    verdict = _choice(row["verdict"], ("PASS", "FAIL", "INCONCLUSIVE"))
    details = [
        f"Baseline {_label(row['baseline_target'])} (run {_label(row['baseline_run'])}). "
        f"Candidate {_label(row['candidate_target'])} "
        + (
            f"(run {_label(row['candidate_run'])})."
            if row.get("candidate_run")
            else "(no run ID recorded)."
        )
    ]
    details += _hashes(
        row,
        (("candidate_sha256", "Candidate configuration"), ("case_sha256", "Case set")),
    )
    if row.get("min_speedup_pct") is not None:
        details.append(
            f"The requested minimum latency reduction was {_fmt(row['min_speedup_pct'])}%."
        )
    metrics = row.get("metrics")
    if metrics is not None:
        m = _object(metrics)
        cases = _count(m["paired_cases"])
        common = _count(m["common_successes"])
        base, candidate = _count(m["baseline_correct"]), _count(m["candidate_correct"])
        if not (common <= min(base, candidate) <= max(base, candidate) <= cases):
            raise ValueError("inconsistent paired case totals")
        details.append(
            f"Graded passes were {base}/{cases} for the baseline and {candidate}/{cases} "
            f"for the candidate. The candidate lost {len(_items(m['lost_cases']))} of the baseline's passes, "
            f"including {len(_items(m['guard_losses']))} on protected cases."
        )
        details.append(
            f"On {common} joint successes, total whole-case time was {_fmt(m['baseline_common_success_s'])} s "
            f"for the baseline and {_fmt(m['candidate_common_success_s'])} s for the candidate. "
            f"Observed latency reduction was {_fmt(m['speedup_pct'])}%."
            if m["speedup_pct"] is not None
            else f"The {common} joint successes have no usable paired latency estimate."
        )
    elif verdict != "INCONCLUSIVE":
        raise ValueError("a completed benchmark verdict requires paired metrics")
    if verdict in ("PASS", "FAIL"):
        _hash(row["candidate_sha256"])
        details += _hashes(
            row,
            (
                ("baseline_runtime_hash", "Baseline runtime"),
                ("active_runtime_hash", "Candidate runtime"),
            ),
        )
        details.append(
            "The benchmark checked the candidate source and active runtime identities. "
            "The receipt gives a descriptive comparison on these cases."
        )
    # Only quote fixed verdict messages. Transport errors can include private URLs or credentials.
    reason = row.get("reason")
    if reason in (
        "protected case lost or task success fell",
        "observed latency reduction did not meet the objective",
        "missing positive whole-case timings on common successes",
        "candidate submitted; awaiting evidence",
        "invalid configuration YAML",
    ):
        details.append(reason[0].upper() + reason[1:].replace("; ", ". ") + ".")
    elif isinstance(reason, str) and reason.startswith(
        "different frozen subject protocol:"
    ):
        details.append("The subjects used different frozen request settings.")
    if verdict == "PASS":
        outcome = (
            "PASS. The candidate met this comparison's quality and latency criteria."
        )
        step = "Check the same configuration on independently reserved tasks and the required baselines before promotion."
    elif verdict == "FAIL":
        outcome = (
            "FAIL. The candidate missed this comparison's quality or latency criteria."
        )
        step = "Retain the baseline for this comparison. Inspect protected losses and joint-success time before another edit."
    else:
        outcome = "INCONCLUSIVE. This receipt does not establish a comparison result."
        step = "Inspect the diagnostic in the run receipt. Reopen the recorded run if it is still active."
    return Finding(
        "Grade a benchmark comparison",
        "Check whether the candidate preserves graded task success and reduces whole-case latency against the named baseline.",
        outcome,
        details,
        step,
    )


def _compare(row):
    split = _choice(row["split"], ("dev", "holdout"))
    disposition = row["disposition"]
    steps = {
        "keep-baseline": "Keep the baseline arm for this comparison.",
        "advance-to-holdout": "Freeze the candidate and evaluate it on the reserved holdout.",
        None: "Review the paired evidence and record a disposition.",
    }
    if disposition is not None:
        _choice(disposition, steps)
    pairs = _object(row["jointly_correct"])
    wins, losses = len(_items(row["wins"])), len(_items(row["losses"]))
    details = [
        f"Baseline ledger record {_label(row['baseline'])}. Candidate ledger record {_label(row['candidate'])}.",
        f"The candidate gained {wins} strict passes, lost {losses} and shared {len(pairs)} successes with the baseline.",
    ]
    if pairs:
        for role in ("baseline", "candidate"):
            seconds = math.fsum(
                _number(pair[role + "_seconds"]) for pair in pairs.values()
            )
            details.append(
                f"Total {role} wall time on joint successes was {_fmt(seconds)} s."
            )
    outcome = (
        f"The {split} comparison records `{disposition}`."
        if disposition
        else f"The {split} comparison has no recorded disposition."
    )
    return Finding(
        "Compare graded ledger arms",
        "Measure changes in strict task success and wall time for matched ledger arms.",
        outcome,
        details,
        steps[disposition],
    )


_RENDER = {
    "recipe-lab-sr-bench-run-v1": _bench,
}


def render_report(evidence: Iterable[dict], *, background: str | None = None) -> str:
    """Render one or more deliberately selected receipts in their supplied order."""
    findings = []
    for row in evidence:
        _object(row)
        schema = row.get("schema")
        if schema is None and row.get("kind") == "comparison":
            render = _compare
        elif isinstance(schema, str) and schema in _RENDER:
            render = _RENDER[schema]
        else:
            raise ValueError("unsupported Recipe Lab report evidence")
        try:
            findings.append(render(row))
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("missing or malformed fields in report evidence") from exc
    if not findings:
        raise ValueError("supply at least one Recipe Lab receipt")
    lines = ["# Recipe Lab report", "", "## Background", ""]
    if background:
        # Indented literal text preserves operator wording without active Markdown.
        lines.extend(
            "    " + _KEY.sub("[redacted]", line)
            for line in background.strip().splitlines()
        )
        lines.append("")
    lines += [" ".join(dict.fromkeys(finding.purpose for finding in findings)), ""]
    if len(findings) > 1:
        last = findings[-1]
        lines += [
            "## Latest supplied result",
            "",
            f"Receipt {len(findings)}. {last.outcome}",
            "",
            f"**Next step.** {last.next_step}",
            "",
            "The receipts below follow the supplied order. Each result applies to its own configuration and cases.",
            "",
        ]
    lines += ["## Recorded results", ""]
    for index, finding in enumerate(findings, 1):
        in_summary = index == len(findings) and len(findings) > 1
        lines += [f"### {index}. {finding.title}"]
        if not in_summary:
            lines += ["", finding.outcome]
        if finding.details:
            lines.append("")
            lines.extend("- " + detail for detail in finding.details)
        if not in_summary:
            lines += ["", f"**Next step.** {finding.next_step}"]
        lines.append("")
    return "\n".join(lines)
