"""Recipe diagnosis: coverage, distributions, threshold reachability (D).

Three defect classes a recipe can carry without any ground truth (sub-issue 7):

- decision coverage: every config decision with its hit count and traffic
  share; zero hits => dead (BUG-3's strong lanes), low share => rare.
  Decisions listed in expect_rare (e.g. security containment) are reported
  but never warned about: a security lane firing on 0.2% of traffic is the
  design working, not a defect.
- signal distributions: per signal_values key, quantiles, a 10-bucket
  histogram over the signal's own range (rule_threshold.signal_range:
  score -> [0, levels-1], noul/choice/set -> [0, 1]) and a range-utilization
  figure ((p95 - p05) / (hi - lo)) that flags compressed score scales before
  any decision goes dead. The sorted sample is stored on the entry, so the
  calibration proposer can recompute any quantile from exactly the evidence
  the diagnosis saw. Signals unknown to the config fall back to their
  observed range and report no utilization.
- threshold reachability: for every decision-rule predicate site
  (rule_threshold.find_threshold_sites), the traffic share satisfying the
  written predicate; zero => unreachable. A site under a NOT branch keeps
  its written predicate as the test (the NOT context is recorded, not
  applied): the defect to report is "this threshold value never separates
  anything", which is exactly the site group the rule-level proposer moves
  together (BUG-3: all four `difficulty gte 2.5` sites, two of them under
  NOT, were stale).

Rows are the unlabeled diagnostic rows the router_replay adapter emits
(records_to_rows): one JSON object per line carrying at least
matched_decision and signal_values. signal_values keys are `<type>:<name>`
(e.g. `decision:difficulty`); choice-type signals store one value per choice
under `decision:<name>:<choice>`, and a rule predicate on such a signal is
evaluated against the strongest choice's confidence.

Output: build_diagnosis returns a `recipe-lab-diagnosis-v1` dict;
write_diagnosis renders it byte-stably to diagnosis.json + diagnosis.md. The
top-level dead_decisions / rare_decisions / unreachable_sites arrays carry
the alerting subset (expect_rare decisions excluded) in the exact shape
rl/04's `next --diagnosis` context consumes, so the two layers converge
without a second format.

Scope discipline: a diagnosis reports phenomena and never judges a recipe;
propose-calibrated turns findings into candidates that still go through the
replay/live tiers like any other candidate.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .contract import ContractError
from .proposers import rule_threshold

DIAGNOSIS_KIND = "recipe-lab-diagnosis-v1"
DEFAULT_RARE_PCT = 1.0
DEFAULT_UTILIZATION_WARN = 0.5
DEFAULT_MIN_SAMPLE = 100
_HIST_BUCKETS = 10


def percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100]) over ascending values.

    The common "linear" method: pos = (n-1)*pct/100, then interpolate between
    the neighbors. Shared with the calibration proposer so a recomputed
    quantile always equals the stored one.
    """
    if not sorted_values:
        raise ContractError("percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (pct / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def _num(value: float) -> float:
    return round(float(value), 6)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def signal_key_name(key: str) -> str:
    """`decision:difficulty` -> `difficulty`; `decision:domain:code` -> `domain`.

    Keys are `<type>:<name>` with an optional third `<choice>` segment for
    per-choice values (verified against the router store, see the adapter
    docstring).
    """
    parts = key.split(":")
    return parts[1] if len(parts) > 1 else key


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _comparator(row: dict[str, Any], signal_name: str) -> float | None:
    """The signal value a rule predicate is tested against on one row.

    Score-type signals keep one numeric value under `decision:<name>`.
    Choice/set signals keep one value per choice under
    `decision:<name>:<choice>`; the predicate is evaluated against the
    strongest choice's confidence.
    """
    values = row.get("signal_values")
    if not isinstance(values, dict):
        return None
    raw = values.get(f"decision:{signal_name}")
    if _numeric(raw):
        return float(raw)
    prefix = f"decision:{signal_name}:"
    candidates = [
        float(v) for k, v in values.items() if str(k).startswith(prefix) and _numeric(v)
    ]
    return max(candidates) if candidates else None


def decision_coverage(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    expect_rare: tuple[str, ...] | list[str] = (),
    rare_pct: float = DEFAULT_RARE_PCT,
) -> dict[str, Any]:
    """Every config decision with its hit count and traffic share.

    status is dead (0 hits), rare (share below rare_pct percent) or ok.
    Entries carry expected_rare for decisions listed in expect_rare; the
    alerting subsets (dead_decisions / rare_decisions) exclude them.
    """
    decisions = [
        decision
        for decision in config.get("routing", {}).get("decisions", [])
        if isinstance(decision, dict)
    ]
    expected = set(expect_rare)
    counts = Counter(str(row.get("matched_decision") or "") for row in rows)
    total = len(rows)
    entries: list[dict[str, Any]] = []
    dead: list[str] = []
    rare: list[dict[str, Any]] = []
    for decision in decisions:
        name = str(decision.get("name", ""))
        hits = counts.get(name, 0)
        share = _num(hits / total * 100.0) if total else 0.0
        status = "dead" if hits == 0 else ("rare" if share < rare_pct else "ok")
        entry = {
            "decision": name,
            "hits": hits,
            "hit_pct": share,
            "status": status,
            "expected_rare": name in expected,
        }
        entries.append(entry)
        if entry["expected_rare"]:
            continue
        if status == "dead":
            dead.append(name)
        elif status == "rare":
            rare.append({"decision": name, "hit_pct": share})
    return {
        "total_decisions": len(decisions),
        "decisions": entries,
        "dead_decisions": dead,
        "rare_decisions": rare,
    }


def _histogram(sample: list[float], lo: float, hi: float) -> dict[str, Any]:
    """10 equal buckets over [lo, hi]; an interior-edge value opens the next
    bucket, and hi itself closes the last one."""
    width = (hi - lo) / _HIST_BUCKETS
    counts = [0] * _HIST_BUCKETS
    for value in sample:
        if width <= 0:
            counts[-1] += 1  # degenerate range: everything in the last bucket
            continue
        index = int((value - lo) / width)
        if index >= _HIST_BUCKETS:
            index = _HIST_BUCKETS - 1
        counts[index] += 1
    edges = [_num(lo + i * width) for i in range(_HIST_BUCKETS + 1)]
    return {"edges": edges, "counts": counts}


def signal_distribution(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Per signal_values key: quantiles, histogram and range utilization.

    One entry per key actually present in the rows (choice signals list one
    entry per choice key). The range comes from rule_threshold.signal_range;
    signals the config does not declare fall back to their observed range and
    carry range: null / range_utilization: null.
    """
    keys: set[str] = set()
    for row in rows:
        values = row.get("signal_values")
        if isinstance(values, dict):
            keys.update(str(k) for k in values)
    signals: list[dict[str, Any]] = []
    for key in sorted(keys):
        sample: list[float] = []
        for row in rows:
            raw = row.get("signal_values")
            if isinstance(raw, dict) and _numeric(raw.get(key)):
                sample.append(float(raw[key]))
        if not sample:
            continue  # the key only ever held non-numeric values
        sample.sort()
        name = signal_key_name(key)
        lo = hi = None
        known = False
        try:
            lo, hi = rule_threshold.signal_range(config, name)
            known = True
        except ContractError:
            lo, hi = sample[0], sample[-1]
        entry: dict[str, Any] = {
            "key": key,
            "signal": name,
            "n": len(sample),
            "min": _num(sample[0]),
            "p05": _num(percentile(sample, 5)),
            "p25": _num(percentile(sample, 25)),
            "p50": _num(percentile(sample, 50)),
            "p75": _num(percentile(sample, 75)),
            "p95": _num(percentile(sample, 95)),
            "max": _num(sample[-1]),
            "mean": _num(sum(sample) / len(sample)),
            # the sorted sample itself: the calibration proposer recomputes
            # arbitrary quantiles from exactly the evidence the diagnosis saw
            "values": [_num(v) for v in sample],
        }
        if known:
            width = hi - lo
            entry["range"] = [_num(lo), _num(hi)]
            entry["range_utilization"] = (
                _num((percentile(sample, 95) - percentile(sample, 5)) / width)
                if width > 0
                else None
            )
        else:
            entry["range"] = None
            entry["range_utilization"] = None
        entry["histogram"] = _histogram(sample, lo, hi)
        signals.append(entry)
    return {"signals": signals}


def threshold_reachability(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Traffic share satisfying every decision-rule predicate site.

    Sites are grouped by (signal, op, value): the set a rule-threshold
    candidate moves together. A group is unreachable only when every site in
    it is; a site whose signal never appears in the traffic is unobserved and
    never makes a group unreachable.
    """
    sites = rule_threshold.find_threshold_sites(config)
    records: list[dict[str, Any]] = []
    for site in sites:
        satisfied = evaluated = 0
        for row in rows:
            value = _comparator(row, site.signal)
            if value is None:
                continue
            evaluated += 1
            if site.op == "gte":
                holds = value >= site.value
            elif site.op == "gt":
                holds = value > site.value
            elif site.op == "lte":
                holds = value <= site.value
            else:
                holds = value < site.value
            if holds:
                satisfied += 1
        records.append(
            {
                "decision": site.decision,
                "signal": site.signal,
                "op": site.op,
                "value": site.value,
                "negated": site.negated,
                "satisfied": satisfied,
                "evaluated": evaluated,
                "satisfied_pct": (
                    _num(satisfied / evaluated * 100.0) if evaluated else None
                ),
                "status": (
                    "unobserved"
                    if not evaluated
                    else ("unreachable" if satisfied == 0 else "reachable")
                ),
            }
        )
    groups: dict[tuple[str, str, float], dict[str, Any]] = {}
    for site, record in zip(sites, records, strict=False):
        key = (site.signal, site.op, site.value)
        group = groups.setdefault(
            key,
            {"signal": site.signal, "op": site.op, "value": site.value, "sites": []},
        )
        group["sites"].append(record)
    group_list = []
    for group in groups.values():
        statuses = {s["status"] for s in group["sites"]}
        if statuses == {"unreachable"}:
            status = "unreachable"
        elif "reachable" in statuses:
            status = "reachable"
        else:
            status = "unobserved"
        group_list.append(
            {
                "signal": group["signal"],
                "op": group["op"],
                "value": group["value"],
                "status": status,
                "sites": group["sites"],
            }
        )
    return {"sites": records, "groups": group_list}


def load_diagnostic_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load unlabeled diagnostic rows (one JSON object per line).

    Rows are what `collect-replays --unlabeled` writes; coverage needs
    matched_decision and the other checks need signal_values. Older
    snapshots without signal_values (pre-adapter) are rejected: the
    distribution and reachability checks would silently report nothing.
    """
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"diagnostic rows file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"diagnostic rows line {lineno} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ContractError(
                    f"diagnostic rows line {lineno} must be a JSON object"
                )
            rows.append(raw)
    if not any(
        isinstance(row.get("signal_values"), dict) and row["signal_values"]
        for row in rows
    ):
        raise ContractError(
            "rows carry no signal_values; re-collect with 'collect-replays --unlabeled' "
            "(pre-adapter snapshots cannot be diagnosed)"
        )
    return rows


def _agentic_evidence(
    rows: list[dict[str, Any]],
    agent_requests: list[Any] | None = None,
) -> dict[str, Any] | None:
    """AG4: the agentic evidence section, or None when the rows carry none.

    Present when any row carries a session phase, an agent marker, a
    session-policy block or a failed lifecycle - legacy single-turn rows
    keep the historical diagnosis shape unchanged. `agent_requests` is the
    client-side sub-agent manifest (Z9) that upgrades the separation check
    from record markers to a request-id join.
    """
    has_evidence = any(
        row.get("phase")
        or row.get("agent_id")
        or isinstance(row.get("session_policy"), dict)
        or str(row.get("lifecycle_state") or "") == "failed"
        for row in rows
    )
    if not has_evidence:
        return None
    return {
        "phase_distributions": phase_distributions(rows),
        "agent_separation": agent_separation(rows, agent_requests=agent_requests),
        "failures": failure_disclosure(rows),
    }


def phase_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """AG4: decision and signal distributions grouped by the arriving
    request's session phase (user_turn / tool_loop / ...).

    BUG-3's lesson: the pilot's difficulty statistics were computed over a
    mixed-phase population, and tool-loop requests run difficulty visibly
    lower - the mixed quantiles describe neither phase. This section only
    splits the same facts per phase; judging remains the proposer's job.
    Rows without a phase (records with no readable request body) stay out.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phase = str(row.get("phase") or "")
        if phase:
            groups.setdefault(phase, []).append(row)
    phases: list[dict[str, Any]] = []
    for phase in sorted(groups):
        phase_rows = groups[phase]
        decisions = Counter(
            str(row.get("matched_decision") or "")
            for row in phase_rows
            if row.get("matched_decision")
        )
        values_by_signal: dict[str, list[float]] = {}
        for row in phase_rows:
            values = row.get("signal_values")
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                # score-type decision signals only (decision:<name>); the
                # per-choice keys (decision:<name>:<choice>) ride the
                # full-signal section unchanged
                if (
                    _numeric(value)
                    and key.startswith("decision:")
                    and key.count(":") == 1
                ):
                    values_by_signal.setdefault(key, []).append(float(value))
        signals: dict[str, Any] = {}
        for key, values in sorted(values_by_signal.items()):
            values.sort()
            signals[key] = {
                "n": len(values),
                "p05": _num(percentile(values, 5)),
                "p50": _num(percentile(values, 50)),
                "p95": _num(percentile(values, 95)),
            }
        phases.append(
            {
                "phase": phase,
                "requests": len(phase_rows),
                "decisions": dict(sorted(decisions.items())),
                "signals": signals,
            }
        )
    return {"phases": phases}


def _identity_missing(row: dict[str, Any]) -> bool:
    """AG4: whether one judged row's protection identity is missing.

    Only rows carrying a session-policy block are judged (the router
    recorded its identity resolution); a missing conversation part or an
    explicit status="missing" counts, rows without the block carry no
    evidence either way.
    """
    policy = row.get("session_policy")
    if not isinstance(policy, dict):
        return False
    identity = policy.get("identity")
    if not isinstance(identity, dict):
        return False
    conversation = identity.get("conversation")
    if isinstance(conversation, dict) and conversation.get("status"):
        return conversation["status"] == "missing"
    return True


def _protection_identity_hash(row: dict[str, Any]) -> str | None:
    """The protection identity hash one judged row resolved to (Z9).

    `identity.memory_key_hash` is the hash the protection actually holds
    state under; None when the row has no trace or no hash (the protection
    never resolved an identity for it).
    """
    policy = row.get("session_policy")
    if not isinstance(policy, dict):
        return None
    identity = policy.get("identity")
    if not isinstance(identity, dict):
        return None
    hash_value = identity.get("memory_key_hash")
    return str(hash_value) if hash_value else None


def agent_separation(
    rows: list[dict[str, Any]],
    *,
    agent_requests: list[Any] | None = None,
) -> dict[str, Any]:
    """AG4: do sub-agent requests keep their own protection identity?

    With a client-side manifest (`agent_requests` - the probe's
    agent-requests-<tag>.json, Z9) the judgment joins those request ids to
    the rows and compares each sub-agent request's `identity.memory_key_hash`
    against the main traffic's: a sub-agent request sharing the main hash -
    or carrying none - is not kept separate (conversation-scoped protection
    would fold the sub-agent's traffic into the main agent's window). A
    manifest id the replays cannot join, or a manifest without any sub-agent
    request, is partial evidence: unverifiable, never asserted either way.

    Without a manifest the record markers are the only evidence: an
    agent-marked request that shares the main conversation's identity - or
    carries none - is not kept separate. When no agent marker exists but
    identities are missing outright, separate retention cannot hold either
    (the AG8 fixture case: the client sends no identity headers at all).
    With no marker and every judged request identified, the check is
    unverifiable - reported as a limitation, never asserted either way.
    """
    judged = [row for row in rows if isinstance(row.get("session_policy"), dict)]
    if agent_requests is not None:
        wanted: list[str] = []
        for entry in agent_requests:
            request_id = entry.get("request_id") if isinstance(entry, dict) else entry
            if str(request_id or "").strip():
                wanted.append(str(request_id).strip())
        if not wanted:
            return {
                "status": "unverifiable",
                "manifest_requests": 0,
                "reason": (
                    "the client manifest names no sub-agent request; there is "
                    "no sub-agent traffic to judge"
                ),
            }
        by_id = {str(row.get("request_id") or ""): row for row in judged}
        joined = [(request_id, by_id.get(request_id)) for request_id in wanted]
        unjoinable = [rid for rid, row in joined if row is None]
        if unjoinable:
            return {
                "status": "unverifiable",
                "manifest_requests": len(wanted),
                "unjoinable_request_ids": unjoinable,
                "reason": (
                    f"{len(unjoinable)} manifest request(s) joined no replay "
                    "row with a protection trace - the evidence is partial"
                ),
            }
        main_hashes = Counter(
            hash_value
            for row in judged
            if str(row.get("request_id") or "") not in set(wanted)
            for hash_value in [_protection_identity_hash(row)]
            if hash_value
        )
        main_hash = main_hashes.most_common(1)[0][0] if main_hashes else None
        if main_hash is None:
            return {
                "status": "unverifiable",
                "manifest_requests": len(wanted),
                "reason": (
                    "no identified non-agent row to compare the sub-agent "
                    "identity against"
                ),
            }
        hashes = {rid: _protection_identity_hash(row) for rid, row in joined}
        shared = sorted(
            rid for rid, hash_value in hashes.items() if hash_value == main_hash
        )
        missing = sorted(
            rid for rid, hash_value in hashes.items() if hash_value is None
        )
        if shared or missing:
            return {
                "status": "not_separate",
                "manifest_requests": len(wanted),
                "shared_identity_request_ids": shared,
                "requests_missing_identity": len(missing),
                "reason": (
                    "sub-agent requests share the main protection identity "
                    f"({len(shared)} request(s)) or carry none ({len(missing)} "
                    "request(s))"
                ),
            }
        return {
            "status": "separate",
            "manifest_requests": len(wanted),
            "reason": (
                "every manifest sub-agent request resolved to a protection "
                "identity distinct from the main traffic"
            ),
        }
    agent_rows = [row for row in judged if row.get("agent_id")]
    missing = [row for row in judged if _identity_missing(row)]
    if agent_rows:
        identities: dict[str, set[str]] = {}
        for row in agent_rows:
            session = row.get("session") if isinstance(row.get("session"), dict) else {}
            conversation = str(
                session.get("conversation_id") or session.get("session_id") or ""
            )
            identities.setdefault(conversation, set()).add(str(row["agent_id"]))
        shared = sorted(
            conversation
            for conversation, markers in identities.items()
            if len(markers) > 1
        )
        agent_missing = sum(1 for row in agent_rows if _identity_missing(row))
        if shared or agent_missing:
            return {
                "status": "not_separate",
                "agent_requests": len(agent_rows),
                "shared_identity_conversations": shared,
                "agent_requests_missing_identity": agent_missing,
                "reason": (
                    "agent-marked requests share the main conversation identity "
                    f"({len(shared)} conversation(s)) or carry none "
                    f"({agent_missing} request(s))"
                ),
            }
        return {
            "status": "separate",
            "agent_requests": len(agent_rows),
            "reason": "every agent-marked request carries its own identity",
        }
    if missing:
        return {
            "status": "not_separate",
            "agent_requests": 0,
            "requests_missing_identity": len(missing),
            "reason": (
                f"protection identity missing on {len(missing)} request(s); "
                "sub-agents cannot keep a separate identity without one"
            ),
        }
    return {
        "status": "unverifiable",
        "agent_requests": 0,
        "reason": (
            "no agent marker in the records (the router does not record agent "
            "headers yet) and every judged request carries an identity"
        ),
    }


def failure_disclosure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """AG4: failed selections disclosed as facts (sub-6 P1).

    A lifecycle_state=failed record is a rejected request (503 under
    selection_rejected); the count and its terminal reasons belong in the
    diagnosis so a protection misconfiguration is visible before any
    proposal is made on top of it.
    """
    failed = [row for row in rows if str(row.get("lifecycle_state") or "") == "failed"]
    reasons = Counter(str(row.get("terminal_reason") or "unknown") for row in failed)
    return {
        "failed_requests": len(failed),
        "terminal_reasons": dict(sorted(reasons.items())),
        "selection_rejected": reasons.get("selection_rejected", 0),
    }


def build_diagnosis(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    rows_path: str = "",
    rows_hash: str = "",
    config_path: str = "",
    config_hash: str = "",
    source: str = "unlabeled replay rows",
    expect_rare: tuple[str, ...] | list[str] = (),
    rare_pct: float = DEFAULT_RARE_PCT,
    utilization_warn: float = DEFAULT_UTILIZATION_WARN,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    agent_requests: list[Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full diagnosis dict (pure function, byte-stable output).

    Alerts collect the actionable findings in a fixed order: dead decisions,
    rare decisions, compressed score ranges, unreachable threshold groups,
    and finally the small-sample warning (traffic representativeness,
    sub-issue 7 §2.4: quantiles are only as good as the traffic behind them).
    `agent_requests` is the client-side sub-agent manifest (Z9) feeding the
    separation join.
    """
    coverage = decision_coverage(
        rows, config, expect_rare=expect_rare, rare_pct=rare_pct
    )
    distribution = signal_distribution(rows, config)
    reachability = threshold_reachability(rows, config)
    agentic = _agentic_evidence(rows, agent_requests=agent_requests)

    alerts: list[dict[str, Any]] = []
    for name in coverage["dead_decisions"]:
        alerts.append({"type": "dead_decision", "decision": name})
    for item in coverage["rare_decisions"]:
        alerts.append(
            {
                "type": "rare_decision",
                "decision": item["decision"],
                "hit_pct": item["hit_pct"],
            }
        )
    for signal in distribution["signals"]:
        utilization = signal["range_utilization"]
        if utilization is not None and utilization < utilization_warn:
            alerts.append(
                {
                    "type": "compressed_range",
                    "signal": signal["signal"],
                    "key": signal["key"],
                    "range_utilization": utilization,
                    "warn_below": _num(utilization_warn),
                }
            )
    unreachable_groups = [
        g for g in reachability["groups"] if g["status"] == "unreachable"
    ]
    for group in unreachable_groups:
        alerts.append(
            {
                "type": "unreachable_threshold",
                "signal": group["signal"],
                "op": group["op"],
                "value": group["value"],
                "sites": len(group["sites"]),
            }
        )
    if len(rows) < min_sample:
        alerts.append(
            {
                "type": "small_sample",
                "samples": len(rows),
                "min_recommended": min_sample,
            }
        )

    return {
        "kind": DIAGNOSIS_KIND,
        "inputs": {
            "rows": rows_path,
            "rows_hash": rows_hash,
            "config": config_path,
            "config_hash": config_hash,
            "source": source,
            "samples": len(rows),
        },
        "decision_coverage": coverage,
        "signal_distribution": distribution,
        "threshold_reachability": reachability,
        **({"agentic": agentic} if agentic is not None else {}),
        # rl/04 `next --diagnosis` compatibility: the alerting subset only
        "dead_decisions": coverage["dead_decisions"],
        "rare_decisions": coverage["rare_decisions"],
        "unreachable_sites": [
            {
                "decision": site["decision"],
                "signal": site["signal"],
                "op": site["op"],
                "threshold": site["value"],
            }
            for group in unreachable_groups
            for site in group["sites"]
        ],
        "alerts": alerts,
    }


def load_diagnosis(path: str | Path) -> dict[str, Any]:
    """Load a diagnosis.json artifact (kind-checked); the calibration
    proposer's input."""
    path = Path(path)
    try:
        diagnosis = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read diagnosis file {str(path)!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"diagnosis file is not valid JSON: {path} ({exc})"
        ) from exc
    if not isinstance(diagnosis, dict) or diagnosis.get("kind") != DIAGNOSIS_KIND:
        raise ContractError(f"{path} is not a {DIAGNOSIS_KIND} artifact")
    return diagnosis


def write_diagnosis(
    diagnosis: dict[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    """Write diagnosis.json and diagnosis.md; same input, same bytes."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diagnosis.json"
    json_path.write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path = out_dir / "diagnosis.md"
    md_path.write_text(render_markdown(diagnosis), encoding="utf-8")
    return json_path, md_path


def _alert_text(alert: dict[str, Any]) -> str:
    kind = alert.get("type")
    if kind == "dead_decision":
        return f"`{alert['decision']}` never fires"
    if kind == "rare_decision":
        return f"`{alert['decision']}` fires on {_fmt(alert['hit_pct'])}% of traffic"
    if kind == "compressed_range":
        return f"`{alert['key']}` uses only {_fmt(alert['range_utilization'])} of its range"
    if kind == "unreachable_threshold":
        return f"`{alert['signal']} {alert['op']} {_fmt(alert['value'])}` is never satisfied ({alert['sites']} site(s))"
    if kind == "small_sample":
        return f"{alert['samples']} samples is below the recommended {alert['min_recommended']}"
    return json.dumps(alert, ensure_ascii=False, sort_keys=True)


def render_markdown(diagnosis: dict[str, Any]) -> str:
    """The human-readable twin of diagnosis.json (deterministic)."""
    inputs = diagnosis.get("inputs", {})
    coverage = diagnosis.get("decision_coverage", {})
    reachability = diagnosis.get("threshold_reachability", {})
    lines = [
        "# Recipe diagnosis report",
        "",
        f"- kind: `{diagnosis.get('kind', '')}`",
        f"- rows: `{inputs.get('rows', '')}` ({inputs.get('rows_hash') or 'n/a'})",
        f"- config: `{inputs.get('config', '')}` ({inputs.get('config_hash') or 'n/a'})",
        f"- source: {inputs.get('source', '')}",
        f"- samples: {inputs.get('samples', 0)}",
        "",
        "## Alerts",
        "",
    ]
    alerts = diagnosis.get("alerts", [])
    if alerts:
        lines.extend(f"- {alert['type']}: {_alert_text(alert)}" for alert in alerts)
    else:
        lines.append("- none")

    lines += [
        "",
        "## Decision coverage",
        "",
        "| decision | hits | share % | status |",
        "|---|---|---|---|",
    ]
    for entry in coverage.get("decisions", []):
        flag = " (expected-rare)" if entry.get("expected_rare") else ""
        lines.append(
            f"| `{entry['decision']}` | {entry['hits']} | {_fmt(entry['hit_pct'])} | {entry['status']}{flag} |"
        )

    lines += ["", "## Signal distributions", ""]
    for signal in diagnosis.get("signal_distribution", {}).get("signals", []):
        lines.append(f"### `{signal['key']}`")
        lines.append("")
        lines.append(
            f"- n: {signal['n']}, min: {_fmt(signal['min'])}, p05: {_fmt(signal['p05'])}, "
            f"p25: {_fmt(signal['p25'])}, p50: {_fmt(signal['p50'])}, p75: {_fmt(signal['p75'])}, "
            f"p95: {_fmt(signal['p95'])}, max: {_fmt(signal['max'])}, mean: {_fmt(signal['mean'])}"
        )
        if signal["range"] is None:
            lines.append("- range: unknown (signal not declared in the config)")
        else:
            lines.append(
                f"- range: [{_fmt(signal['range'][0])}, {_fmt(signal['range'][1])}], "
                f"range_utilization: {_fmt(signal['range_utilization'])}"
            )
        edges = signal["histogram"]["edges"]
        counts = signal["histogram"]["counts"]
        lines += ["", "| from | to | count |", "|---|---|---|"]
        for index, count in enumerate(counts):
            lines.append(
                f"| {_fmt(edges[index])} | {_fmt(edges[index + 1])} | {count} |"
            )
        lines.append("")

    lines += [
        "## Threshold reachability",
        "",
        "| decision | predicate | negated | satisfied | evaluated | share % | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for site in reachability.get("sites", []):
        predicate = f"{site['signal']} {site['op']} {_fmt(site['value'])}"
        lines.append(
            f"| `{site['decision']}` | `{predicate}` | {str(site['negated']).lower()} "
            f"| {site['satisfied']} | {site['evaluated']} | {_fmt(site['satisfied_pct'])} | {site['status']} |"
        )
    lines += ["", "Groups (a rule-threshold candidate moves one group together):", ""]
    groups = reachability.get("groups", [])
    if groups:
        for group in groups:
            lines.append(
                f"- `{group['signal']}` `{group['op']}` `{_fmt(group['value'])}`: {group['status']} "
                f"({len(group['sites'])} site(s))"
            )
    else:
        lines.append("- none (no rule-level predicate sites in the config)")

    agentic = diagnosis.get("agentic")
    if agentic:
        lines += ["", "## Agentic session evidence", ""]
        phases = (agentic.get("phase_distributions") or {}).get("phases", [])
        if phases:
            lines += [
                "### Decisions by session phase",
                "",
                "| phase | requests | decisions |",
                "|---|---|---|",
            ]
            for entry in phases:
                decisions = ", ".join(
                    f"`{name}`={count}" for name, count in entry["decisions"].items()
                )
                lines.append(
                    f"| {entry['phase']} | {entry['requests']} | {decisions or '-'} |"
                )
            signal_keys = sorted(
                {key for entry in phases for key in (entry.get("signals") or {})}
            )
            for key in signal_keys:
                lines += [
                    "",
                    f"### `{key}` by session phase",
                    "",
                    "| phase | n | p05 | p50 | p95 |",
                    "|---|---|---|---|---|",
                ]
                for entry in phases:
                    stats = (entry.get("signals") or {}).get(key)
                    if stats is None:
                        continue
                    lines.append(
                        f"| {entry['phase']} | {stats['n']} | {_fmt(stats['p05'])} "
                        f"| {_fmt(stats['p50'])} | {_fmt(stats['p95'])} |"
                    )
        separation = agentic.get("agent_separation") or {}
        lines += [
            "",
            "### Sub-agent separate retention",
            "",
            f"- status: **{separation.get('status', 'unverifiable')}**",
            f"- {separation.get('reason', '')}",
        ]
        failures = agentic.get("failures") or {}
        reason_text = ", ".join(
            f"`{reason}`={count}"
            for reason, count in failures.get("terminal_reasons", {}).items()
        )
        lines += [
            "",
            "### Failed selections",
            "",
            f"- failed requests: {failures.get('failed_requests', 0)} "
            f"(selection_rejected: {failures.get('selection_rejected', 0)})",
            f"- terminal reasons: {reason_text or '-'}",
        ]
    return "\n".join(lines) + "\n"
