"""Render selected Recipe Lab receipts without reading configs, traces or task text."""

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

# A backend/model identifier longer than this is treated as untrusted text.
MAX_LABEL_LENGTH = 128


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
        or len(value) > MAX_LABEL_LENGTH
        or _KEY.search(value)
        or re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", value)
    ):
        return "(identifier omitted)"
    return f"`{value}`"


def _probe(row):
    step = _object(row["next"])
    action = _label(step["action"])
    details = _hashes(step, (("candidate_sha256", "Planned candidate"),))
    if "public_probe" not in step:
        return Finding(
            "Plan a development check",
            "Choose the next routing check using recorded development traffic and the candidate configurations.",
            "A development action was planned.",
            details,
            f"Follow the planned action {action}.",
        )
    probe = _object(step["public_probe"])
    meanings = {
        "fixture_pass": "The public probe passed with fixture backends.",
        "real_pass": "The public probe passed with real backends.",
        "failed": "The public probe failed.",
        "incomplete": "The public probe has incomplete evidence.",
    }
    status = _choice(probe["status"], meanings)
    if status.endswith("_pass") and probe["kind"] != status.split("_")[0]:
        raise ValueError("probe mode disagrees with its status")
    details += _hashes(probe, (("receipt_sha256", "Probe receipt"),))
    details.append(f"The probe recorded {len(_items(probe['issues']))} issues.")
    for key, label in (
        ("deployment_config_verified", "Deployment configuration identity"),
        ("tool_roundtrip_verified", "Tool round trip"),
        ("context_fallback_observed", "Context fallback"),
    ):
        if key in probe:
            details.append(
                f"{label} was "
                + ("confirmed." if probe[key] is True else "unconfirmed.")
            )
    details.append(
        "These checks cover gateway routing and capability handling. Task quality and latency require graded cases."
    )
    return Finding(
        "Validate gateway behavior",
        "Check the candidate's routing boundary and capability handling against a public-gateway probe.",
        meanings[status],
        details,
        f"Follow the planner action {action}.",
    )


def _decision(row):
    status = _choice(
        row["decision"], ("advance", "drop", "revise", "defer", "run", "not_enrolled")
    )
    _choice(row["split"], ("dev",))
    actions = {
        "advance": "Freeze this candidate for an independent evaluation.",
        "drop": "Keep the parent and review the regressed development control.",
        "revise": "Review the observed routing changes and paired times before another edit.",
        "defer": "Resolve the outstanding evidence requirement before funding another trial.",
        "run": "Finish the scheduled development comparison.",
        "not_enrolled": "Enroll the experiment card before dispatch.",
    }
    details = _hashes(
        row,
        (
            ("parent_config_sha256", "Parent configuration"),
            ("candidate_sha256", "Candidate configuration"),
            ("card_sha256", "Experiment card"),
        ),
    )
    if "cost" in row:
        cost = row["cost"]
        details.append(
            f"The study spent {_count(cost['spent_episodes'])} of {_count(cost['budget_episodes'])} episodes, "
            f"including {_count(cost['invalid_attempts'])} invalid attempts."
        )
    reason = row.get("reason")
    if reason in (
        "no observed target predicate change",
        "episode budget exhausted before complete pairs",
        "the fresh guard baseline did not reproduce",
        "opposite paired quality outcomes at equal pass counts",
        "paired development comparison",
    ):
        details.append(f"The recorded reason was “{reason}”.")
    for key, label in (("guard_task", "Guard"), ("target_task", "Target")):
        pair = row.get("by_task", {}).get(row.get(key), {}).get("paired")
        if pair:
            details.append(
                f"{label} strict passes were {_count(pair['baseline_passes'])} for the parent and "
                f"{_count(pair['candidate_passes'])} for the candidate. There were {_count(pair['joint_successes'])} "
                f"joint successes with {_fmt(pair['joint_seconds_saved'])} s saved."
            )
    return Finding(
        "Review a scheduled development experiment",
        "Decide whether to advance a candidate using matched parent and candidate episodes on a target and a guard task.",
        f"The recorded development decision is `{status}`.",
        details,
        actions[status],
    )


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


def _draft(row):
    _choice(row["status"], ("statically_validated",))
    config = _hash(row["sha256"]["config.yaml"])
    details = [
        f"Source template {_label(row['template']['bundle'])} / {_label(row['template']['recipe'])}. "
        f"The public model is {_label(row['public_model'])}.",
        f"Deliver the generated `config.yaml` with SHA-256 {config}.",
    ]
    for lane in ("fast", "reasoning", "tools"):
        refs = _items(row["bindings"][lane])
        if not refs:
            raise ValueError("draft lane is missing its bindings")
        models = " then ".join(_label(ref["model"]) for ref in refs)
        details.append(f"The {lane} lane considers {models} in that order.")
    details.append(
        f"The CLI offline check and native Router validation completed. "
        f"The native Router warning count was {_count(row['router_warning_count'])}."
    )
    details.append(
        "These checks establish configuration validity. Gateway behavior and task quality remain unmeasured by this receipt."
    )
    return Finding(
        "Generate a recipe",
        "Derive a two-backend routing configuration from the installed Speed template and check it before startup.",
        "The draft passed static configuration validation.",
        details,
        "Start the packaged configuration and probe both backends, a tool round trip and an expected refusal. "
        "Measure task success and latency on representative work.",
    )


def _depth(row):
    status = _choice(row["status"], ("proposed", "keep-parent", "inconclusive"))
    details = [
        f"The requested development comparison uses run {_label(row['dev_run'])}, "
        f"parent target {_label(row['parent_target'])} and fast target {_label(row['fast_target'])}."
    ]
    if "observations" in row:
        roles = Counter(
            _choice(item["role"], ("move", "protect", "skip"))
            for item in _items(row["observations"])
        )
        details.append(
            f"The recorded opening requests contained {roles['move']} possible moves, "
            f"{roles['protect']} protected cases and {roles['skip']} skipped cases."
        )
    details += _hashes(row, (("parent_sha256", "Parent configuration"),))
    if status == "proposed":
        details += [
            f"The {_label(row['signal'])} threshold changes from {_fmt(row['before'])} to {_fmt(row['after'])}.",
            f"Proposed configuration SHA-256 {_hash(row['candidate_sha256'])}.",
        ]
        outcome = "The development evidence identified a candidate threshold."
        step = "Deploy the identified candidate on a test endpoint and grade it against the parent."
    elif status == "keep-parent":
        outcome = "Keep the parent. The observed depth scores offered no separating threshold."
        step = "Use another development sample or investigate a different routing decision."
    else:
        outcome = "The depth proposal is inconclusive."
        step = "Review the diagnostic in the proposal receipt and restore the missing development evidence."
    return Finding(
        "Propose a depth change",
        "Find a reasoning threshold that can move development requests to the fast backend while preserving protected requests.",
        outcome,
        details,
        step,
    )


_RENDER = {
    "recipe-lab-v05-speed-draft-v1": _draft,
    "recipe-lab-development-plan-v1": _probe,
    "recipe-lab-decision-summary-v1": _decision,
    "recipe-lab-sr-bench-run-v1": _bench,
    "recipe-lab-sr-bench-depth-v1": _depth,
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
