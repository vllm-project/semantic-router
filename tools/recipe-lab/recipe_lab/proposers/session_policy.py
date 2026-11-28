"""Session-policy proposer (AG6): cause-keyed one-variable candidates.

Z6: the rules follow the real router's protection mechanism (upstream
#5567) - the tool loop is a hard lock independent of the tuning knobs,
a drifted decision releases the lock (traced `decision_drift`, in-design),
and `switch_margin` / `stability_weight` only widen the hold on ordinary
(out-of-lock) scoring. Each candidate changes exactly one variable and is
only proposed when its cause asks for it:

- unprotected tool-loop switches (the summary's tool-loop total minus the
  drift releases - switches whose protection never ran) -> identity
  candidates: configure the protection identity headers (the W1/W2 class)
  so the lock can engage at all
- model flips outside the lock (route-replay summary) -> raise
  `switch_margin` and `stability_weight`; both act on ordinary scoring,
  never on the lock
- sub-agent separation reported broken (the AG4 diagnosis's
  `agent_separation`) -> add the agent identity header

Causes that must NOT become protection knobs come back as handoffs
(`session_policy_handoffs`, not config candidates): drift releases over
the per-task ceiling point at threshold calibration (#5601's
propose-calibrated, the tool-loop phase score distribution), rejected
selections point at the candidate set's relationship to the lock, and
unprotected switches under already-configured headers mean the client is
not sending them.

Evidence the proposer never saw produces no candidate - a proposal always
names the trigger that asked for it. The candidate configs then go through
the AG3 route-replay gates before anything reaches the live tier.

Pure functions, no I/O, deterministic: same inputs -> same candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from recipe_lab.contract import DEFAULT_AGENTIC_GATES

# The agent identity header Claude Code natively sends on delegated
# subagents (AG0-W2).
AGENT_HEADER = "x-claude-code-agent-id"

# The identity headers the ecosystem sends (W1: sr-bench's engine, W2:
# Claude Code) - an identity candidate maps the protection onto them.
IDENTITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("session", "x-session-id"),
    ("conversation", "x-conversation-id"),
)

# The raised values the proposer moves a knob to; each doubles as the
# exhaustion check ("already at the raised value proposes nothing").
STABILITY_WEIGHT_RAISED = 2.0
SWITCH_MARGIN_RAISED = 0.1
# The out-of-lock trigger: enough absolute flips, or a high per-session
# mean, indicate lane flapping that the hold knobs (ordinary scoring)
# can address - the tool loop's hard lock never needed them.
SWITCH_MARGIN_MIN_FLIPS = 2
SWITCH_MARGIN_MEAN_ALERT = 0.5


@dataclass(frozen=True)
class SessionPolicyCandidate:
    variable: str
    old: Any
    new: Any
    trigger: dict[str, Any]
    ref: str
    config: dict[str, Any]

    def summary(self) -> str:
        return (
            f"session policy: {self.variable} {self.old!r} -> {self.new!r} "
            f"(trigger: {self.trigger.get('reason', '')})"
        )


def _protection(config: dict[str, Any]) -> dict[str, Any]:
    learning = (
        config.get("global", {}).get("router", {}).get("learning")
        if isinstance(config.get("global"), dict)
        else None
    )
    protection = learning.get("protection") if isinstance(learning, dict) else None
    if not isinstance(protection, dict):
        raise ValueError(
            "config carries no global.router.learning.protection section; "
            "a session-policy candidate needs one to change"
        )
    return protection


def _tuning(config: dict[str, Any]) -> dict[str, Any]:
    protection = _protection(config)
    tuning = protection.get("tuning")
    if not isinstance(tuning, dict):
        tuning = {}
        protection["tuning"] = tuning
    return tuning


def _ref(variable: str, new: Any) -> str:
    payload = json.dumps({"variable": variable, "new": new}, sort_keys=True)
    return (
        "config-rev:session-policy:" + hashlib.sha256(payload.encode()).hexdigest()[:12]
    )


def _candidate(
    base: dict[str, Any], variable: str, old: Any, new: Any, trigger: dict[str, Any]
) -> SessionPolicyCandidate:
    return SessionPolicyCandidate(
        variable=variable,
        old=old,
        new=new,
        trigger=trigger,
        ref=_ref(variable, new),
        config=copy.deepcopy(base),
    )


def _unprotected_switches(summary: dict[str, Any]) -> float:
    """Tool-loop switches whose protection never ran (Z5's cause split):
    the tool-loop total minus the drift releases the traces accounted for."""
    total = summary.get("tool_loop_switches") or 0
    drift = summary.get("decision_drift_switches") or 0
    return total - drift


def _out_of_lock_switches(summary: dict[str, Any]) -> float:
    """Model flips outside the tool loop - the lane the hold knobs govern."""
    return (summary.get("model_switches") or 0) - (
        summary.get("tool_loop_switches") or 0
    )


def _out_of_lock_flapping(summary: dict[str, Any]) -> bool:
    """Enough out-of-lock flips, or a high out-of-lock per-session mean, to
    ask the hold knobs for a wider hold. The summary's mean counts every
    switch; the trigger rate is cause-split like the flips."""
    sessions = summary.get("sessions") or 0
    flips = _out_of_lock_switches(summary)
    mean = flips / sessions if sessions > 0 else None
    return flips >= SWITCH_MARGIN_MIN_FLIPS or (
        mean is not None and mean > SWITCH_MARGIN_MEAN_ALERT
    )


def _missing_identity_headers(config: dict[str, Any]) -> dict[str, str]:
    """The identity headers the protection would need but the config does
    not map yet (the W1/W2 defaults the ecosystem actually sends)."""
    protection = _protection(config)
    identity = protection.get("identity")
    headers = (
        identity.get("headers")
        if isinstance(identity, dict) and isinstance(identity.get("headers"), dict)
        else {}
    )
    return {
        name: default
        for name, default in IDENTITY_HEADERS
        if not str(headers.get(name) or "").strip()
    }


def _identity_candidate(base, summary) -> SessionPolicyCandidate | None:
    missing = _missing_identity_headers(base)
    if _unprotected_switches(summary) <= 0 or not missing:
        return None  # no cause, or the client is not sending them (handoff)
    candidate = _candidate(
        base,
        "identity.headers",
        None,
        missing,
        {
            "reason": "unprotected_tool_loop_switches",
            "unprotected_tool_loop_switches": _unprotected_switches(summary),
            "note": "switches the protection never saw - map the identity "
            "headers the client sends so the lock can engage (W1/W2)",
        },
    )
    protection = _protection(candidate.config)
    identity = protection.get("identity")
    if not isinstance(identity, dict):
        identity = {}
        protection["identity"] = identity
    headers = identity.get("headers")
    if not isinstance(headers, dict):
        headers = {}
        identity["headers"] = headers
    headers.update(missing)
    return candidate


def _stability_candidate(base, tuning, summary) -> SessionPolicyCandidate | None:
    if not _out_of_lock_flapping(summary):
        return None
    current = tuning.get("stability_weight", 1.0)
    if current >= STABILITY_WEIGHT_RAISED:
        return None  # already at the raised weight; the knob is exhausted
    candidate = _candidate(
        base,
        "stability_weight",
        current,
        STABILITY_WEIGHT_RAISED,
        {
            "reason": "model_switches",
            "out_of_lock_switches": _out_of_lock_switches(summary),
            "note": "ordinary (out-of-lock) scoring flips the lane; a raised "
            "stability weight widens the hold - the tool loop's hard lock "
            "never needed it",
        },
    )
    _tuning(candidate.config)["stability_weight"] = STABILITY_WEIGHT_RAISED
    return candidate


def _switch_margin_candidate(base, tuning, summary) -> SessionPolicyCandidate | None:
    if not _out_of_lock_flapping(summary):
        return None
    current = tuning.get("switch_margin", 0.05)
    if current >= SWITCH_MARGIN_RAISED:
        return None  # the margin is already at its proposed value
    candidate = _candidate(
        base,
        "switch_margin",
        current,
        SWITCH_MARGIN_RAISED,
        {
            "reason": "model_switches",
            "out_of_lock_switches": _out_of_lock_switches(summary),
            "note": "sessions flip outside the lock; a wider margin holds the lane",
        },
    )
    _tuning(candidate.config)["switch_margin"] = SWITCH_MARGIN_RAISED
    return candidate


def _agent_header_candidate(base, diagnosis) -> SessionPolicyCandidate | None:
    separation = ((diagnosis or {}).get("agentic") or {}).get("agent_separation") or {}
    if separation.get("status") != "not_separate":
        return None
    protection = _protection(base)
    identity = protection.get("identity")
    if not isinstance(identity, dict):
        identity = {}
        protection["identity"] = identity
    headers = identity.get("headers")
    if not isinstance(headers, dict):
        headers = {}
        identity["headers"] = headers
    if headers.get("agent"):
        return None  # the agent header is already configured
    candidate = _candidate(
        base,
        "identity.headers.agent",
        None,
        AGENT_HEADER,
        {
            "reason": "agent_separation",
            "evidence": separation.get("reason", ""),
            "note": "sub-agents are not kept separate; hold per-agent identity",
        },
    )
    _protection(candidate.config)["identity"]["headers"]["agent"] = AGENT_HEADER
    return candidate


def propose_session_policy(
    config: dict[str, Any],
    *,
    diagnosis: dict[str, Any] | None = None,
    replay_summary: dict[str, Any] | None = None,
) -> list[SessionPolicyCandidate]:
    """Cause-keyed one-variable session-policy candidates (Z6).

    `replay_summary` is the AG3 route-replay evidence summary (switches by
    cause, rejected selections); `diagnosis` is the AG4 diagnosis artifact
    whose `agentic.agent_separation` reports sub-agent retention. Evidence
    absent produces no candidate; a knob already at its raised value
    produces no candidate either (nothing left to change). Causes that must
    not become protection knobs (drift releases, rejected selections,
    client-side identity) come back from `session_policy_handoffs` instead.
    """
    _protection(config)  # fail fast on a config without a protection section
    tuning = _tuning(config)
    summary = replay_summary or {}
    candidates = []
    for candidate in (
        _identity_candidate(config, summary),
        _stability_candidate(config, tuning, summary),
        _switch_margin_candidate(config, tuning, summary),
        _agent_header_candidate(config, diagnosis),
    ):
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def session_policy_handoffs(
    config: dict[str, Any], *, replay_summary: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Cause-keyed actions that must not become protection knobs (Z6).

    Drift releases are in-design (#5567): over the per-task ceiling they
    point at threshold calibration, not a protection knob. Rejected
    selections point at the candidate set's relationship to the lock.
    Unprotected switches under configured identity headers mean the client
    is not sending them (W1/W2 class). Handoffs ride the proposal report;
    they never produce config candidates.
    """
    summary = replay_summary or {}
    handoffs: list[dict[str, Any]] = []
    sessions = summary.get("sessions") or 0
    drift = summary.get("decision_drift_switches") or 0
    drift_ceiling = DEFAULT_AGENTIC_GATES["max_decision_drift_switches_per_task"]
    if sessions > 0 and drift / sessions > drift_ceiling:
        handoffs.append(
            {
                "reason": "decision_drift_rate",
                "decision_drift_switches": drift,
                "sessions": sessions,
                "note": "drift releases are in-design (#5567): recalibrate the "
                "drift threshold from the tool-loop phase score distribution "
                "(propose-calibrated)",
            }
        )
    if _unprotected_switches(summary) > 0 and not _missing_identity_headers(config):
        handoffs.append(
            {
                "reason": "unprotected_tool_loop_switches",
                "unprotected_tool_loop_switches": _unprotected_switches(summary),
                "note": "identity headers are configured but the client is not "
                "sending them - fix the client's identity headers (W1/W2)",
            }
        )
    rejected = summary.get("selection_rejected") or 0
    if rejected > 0:
        handoffs.append(
            {
                "reason": "selection_rejected",
                "selection_rejected": rejected,
                "note": "check the candidate set's relationship to the lock - is "
                "the locked model still among the drifted decision's "
                "candidates? - not min_turns_before_switch",
            }
        )
    return handoffs


def write_candidates(
    candidates: list[SessionPolicyCandidate], output_dir: str | Path
) -> list[Path]:
    """Write each candidate config plus its sidecar metadata; returns config paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, candidate in enumerate(candidates):
        path = out_dir / f"candidate-{index:03d}-{candidate.variable}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(candidate.config, fh, allow_unicode=True, sort_keys=False)
        sidecar = {
            "kind": "config",
            "ref": candidate.ref,
            "change_summary": candidate.summary(),
            "provenance": {
                "proposer": "session-policy",
                "variable": candidate.variable,
                "old": candidate.old,
                "new": candidate.new,
                "trigger": candidate.trigger,
                "insight_refs": [],
            },
        }
        path.with_suffix(".candidate.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths
