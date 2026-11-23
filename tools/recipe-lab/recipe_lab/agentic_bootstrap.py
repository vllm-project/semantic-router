"""Agentic cold start: profile aggregation and lane assignment (AG5).

The cold-start path for agentic workloads mirrors the MCQ one (C5): a
per-model profile feeds a deterministic lane assignment over the agent
recipe template, and every lane's choice carries its rule and the profile
evidence behind it in a sidecar.

- The profile (`recipe-lab-agentic-profile-v1`) aggregates one single-model
  sr-bench run per model (the AG1 protocol): task success rate with its
  interval, per-task actual and cache-neutral cost, calls per task,
  latency and the truncation rate. Unknown fields stay None - missing
  evidence is never counted as zero.
- Lane assignment reuses the template path's shared selection core
  (`bindings.plan_lane`, Z10): the agentic profile measures models, not
  lanes, so every template lane is an *unmeasured* lane in the sub-issue-8
  §2.2 sense - the candidates are capability-filtered, ranked by the
  lambda-adjusted overall score, and the sidecar marks every choice
  "default rule (unmeasured lane)" exactly as the built-in template path
  does. The hand-rolled per-lane rule zoo this module once carried had no
  lane evidence behind it and disagreed with the template path on
  capability filtering, cost ranking and the untested marking.
- The privacy and security lanes are not a ranking decision: they are
  pinned to the contract's local models with `adaptations.mode: bypass`
  (traffic that must never leave the deployment).
- Session protection is written per client: claude-code protects the
  whole declared agent session keyed by the verified client headers
  (x-claude-code-session-id, agent: x-claude-code-agent-id, from AG0-W2);
  every other client protects one conversation keyed by the neutral
  headers. Adaptation is disabled, the progress gate starts in observe
  mode, candidate requirements demand declared capabilities (tool use is
  a model-card capability), and the decision model stays pinned to the
  template's deployment.

Pure functions, no I/O, deterministic: same inputs -> same config bytes.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .bindings import plan_lane

PROFILE_KIND = "recipe-lab-agentic-profile-v1"
BOOTSTRAP_KIND = "recipe-lab-agentic-bootstrap-v1"

CLIENTS = ("claude-code", "openai-chat", "generic")
# AG0-W2: Claude Code natively sends x-claude-code-session-id and (on
# delegated subagents) x-claude-code-agent-id; it never sends the neutral
# headers. Every other client uses the neutral pair.
CLIENT_IDENTITY_HEADERS = {
    "claude-code": {
        "scope": "session",
        "session": "x-claude-code-session-id",
        "agent": "x-claude-code-agent-id",
    },
    "openai-chat": {
        "scope": "conversation",
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    },
    "generic": {
        "scope": "conversation",
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    },
}

# template decision -> lane kind. "keep" decisions are rewritten by nobody
# (the omni visual-language lane is a hard modality route, not a profile
# choice); "local" decisions are pinned, not ranked.
LANE_DECISIONS: dict[str, str] = {
    "domain_code_complex": "ranked",
    "complex_general": "ranked",
    "domain_code": "ranked",
    "domain_business": "ranked",
    "medium_general": "ranked",
    "domain_stem_research": "ranked",
    "domain_legal_health": "ranked",
    "simple_math_fast_path": "ranked",
    "simple_general": "ranked",
    "local_security_containment": "local",
    "local_privacy_policy": "local",
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def profile_from_reports(reports: Sequence[Any]) -> dict[str, Any]:
    """Aggregate per-model single-target sr-bench run reports (AG1's
    protocol) into the agentic profile.

    Each report contributes the arm a single-model run serves: the `mom`
    target when present, else the last target with an accuracy (the same
    pinning the live tier uses). Fields a report does not carry stay None
    and are flagged in the model entry's `missing` list - the lane
    assignment treats missing cost as "cannot be price-compared", never as
    free.
    """
    models: list[dict[str, Any]] = []
    for index, report in enumerate(reports, 1):
        summary = report.get("summary") if isinstance(report, dict) else None
        targets = summary.get("targets") if isinstance(summary, dict) else None
        entries = [t for t in targets or [] if isinstance(t, dict)]
        target = next((t for t in entries if t.get("kind") == "mom"), None)
        if target is None:
            for candidate in reversed(entries):
                if _number(candidate.get("accuracy")) is not None:
                    target = candidate
                    break
        if target is None:
            raise ValueError(f"report {index} carries no target with an accuracy")
        total = target.get("total")
        total = total if isinstance(total, int) and total > 0 else None
        accuracy = _number(target.get("accuracy"))
        diagnostics = target.get("output_diagnostics")
        planned = (
            diagnostics.get("planned_cases") if isinstance(diagnostics, dict) else None
        )
        output_limit = (
            diagnostics.get("output_limit_cases")
            if isinstance(diagnostics, dict)
            else None
        )
        calls = (
            diagnostics.get("subject_calls") if isinstance(diagnostics, dict) else None
        )
        call_total = calls.get("total") if isinstance(calls, dict) else None
        spend = _number(target.get("total_spend_usd"))
        neutral = _number(target.get("cache_neutral_cost_usd"))
        missing: list[str] = []
        if accuracy is None:
            missing.append("success_rate")
        cost_per_task = spend / total if spend is not None and total else None
        if cost_per_task is None:
            missing.append("cost_per_task_usd")
        neutral_per_task = neutral / total if neutral is not None and total else None
        if neutral_per_task is None:
            missing.append("cache_neutral_cost_per_task_usd")
        mean_calls = (
            call_total / total if isinstance(call_total, int) and total else None
        )
        if mean_calls is None:
            missing.append("mean_calls_per_task")
        truncation = (
            output_limit / planned
            if isinstance(output_limit, int)
            and isinstance(planned, int)
            and planned > 0
            else None
        )
        if truncation is None:
            missing.append("truncation_rate")
        ci = target.get("accuracy_ci95")
        models.append(
            {
                "model": str(target.get("id") or f"report-{index}"),
                "tasks": total,
                "success_rate": accuracy,
                "ci95": [c for c in (ci or []) if _number(c) is not None] or None,
                "cost_per_task_usd": cost_per_task,
                "cache_neutral_cost_per_task_usd": neutral_per_task,
                "mean_calls_per_task": mean_calls,
                "mean_latency_p50_s": _number(target.get("latency_p50_s")),
                "truncation_rate": truncation,
                "missing": sorted(missing),
            }
        )
    return {"kind": PROFILE_KIND, "models": models}


def validate_profile(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Kind- and shape-check a profile; returns its model entries."""
    if not isinstance(profile, Mapping) or profile.get("kind") != PROFILE_KIND:
        raise ValueError(
            f"profile must be a {PROFILE_KIND} artifact, got {profile.get('kind')!r}"
            if isinstance(profile, Mapping)
            else f"profile must be a {PROFILE_KIND} artifact"
        )
    models = profile.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("profile carries no models")
    for entry in models:
        if not isinstance(entry, Mapping) or not str(entry.get("model") or "").strip():
            raise ValueError("profile model entries need a model name")
        rate = _number(entry.get("success_rate"))
        if rate is None or not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"profile model {entry.get('model')!r} needs a success_rate in [0, 1]"
            )
    return list(models)


def _binding_stats(profile: Mapping[str, Any]) -> dict[str, Any]:
    """The profile in the shape the shared selection core reads (Z10).

    `bindings.plan_lane` consumes the C3 profile's calib stats. The agentic
    profile measures models, not lanes, so the adapter synthesizes an
    empty-labels calib block: every lane stays unmeasured and the ranking
    reads the models' overall success rate and per-task cost.
    """
    return {
        "model_split_stats": {
            "calib": {
                str(entry["model"]): {
                    "total": {
                        "accuracy": _number(entry.get("success_rate")),
                        "cost_per_question": _number(entry.get("cost_per_task_usd")),
                        "n": entry.get("tasks"),
                    },
                    "labels": {},
                }
                for entry in validate_profile(profile)
            }
        }
    }


def _profile_models(profile: Mapping[str, Any]) -> list[Any]:
    """The profiled models as selection candidates (name + capabilities).

    The agentic profile carries no capability cards; the shared capability
    filter sees an empty requirement for the agent recipe's decisions, so
    every profiled model is eligible - the evidence-backed candidate set is
    exactly the profiled one.
    """
    return [
        SimpleNamespace(name=str(entry["model"]), capabilities=None)
        for entry in validate_profile(profile)
    ]


def _set_lane_models(decision: dict[str, Any], model: str) -> None:
    decision["modelRefs"] = [{"model": model, "use_reasoning": False}]


def bootstrap_agentic_config(
    template: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    client: str,
    local_models: Sequence[str] = (),
    lam: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite the agent recipe template from the profile (pure function).

    Returns (config, sidecar). The ranked lanes go through the shared
    selection core (`bindings.plan_lane`, unmeasured-lane semantics); the
    sidecar records every lane's choice with its basis and scores, the
    client identity contract and the protection settings, so the
    candidate's provenance can answer "why is this model on this lane"
    without re-deriving anything.
    """
    if client not in CLIENTS:
        raise ValueError(f"client must be one of {list(CLIENTS)}, got {client!r}")
    stats = _binding_stats(profile)
    candidates = _profile_models(profile)
    config = copy.deepcopy(dict(template))
    decisions, decisions_parent = _decisions_of(config)
    if not isinstance(decisions, list):
        raise ValueError("template carries no decisions list")
    reasons: dict[str, Any] = {}
    rewritten = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        name = str(decision.get("name") or "")
        lane = LANE_DECISIONS.get(name)
        if lane is None:
            rewritten.append(decision)
            continue
        if lane == "local":
            model = _pin_local_model(local_models)
            reason: dict[str, Any] = {
                "model": model,
                "basis": "contract-declared local model (privacy/security bypass)",
            }
        else:
            plan = plan_lane(
                lane=name,
                template_lane={"immediate": False, "minimum_candidates": 1},
                models=candidates,
                profile=stats,
                lam=lam,
                min_lane_cases=1,
                lane_counts={name: 0},
            )
            model = plan["selected"][0]
            reason = {
                "model": model,
                "basis": plan["basis"],
                "untested": True,
                "scores": plan["scores"],
            }
        _set_lane_models(decision, model)
        if lane == "local":
            adaptations = decision.get("adaptations")
            if not isinstance(adaptations, dict):
                adaptations = {}
            adaptations["mode"] = "bypass"
            decision["adaptations"] = adaptations
        rewritten.append(decision)
        reasons[name] = {"lane": lane, **reason}
    decisions_parent["decisions"] = rewritten

    identity = CLIENT_IDENTITY_HEADERS[client]
    headers: dict[str, str] = {"session": identity["session"]}
    if "conversation" in identity:
        headers["conversation"] = identity["conversation"]
    if "agent" in identity:
        headers["agent"] = identity["agent"]

    global_section = config.get("global")
    if not isinstance(global_section, dict):
        raise ValueError("template carries no global section")
    router = global_section.get("router")
    if not isinstance(router, dict):
        raise ValueError("template carries no global.router section")
    learning = router.get("learning")
    if not isinstance(learning, dict):
        raise ValueError("template carries no global.router.learning section")
    # the protection below is dead configuration under a learning-disabled
    # template (the preflight sees learning disabled and never engages)
    learning["enabled"] = True
    adaptation = learning.get("adaptation")
    if not isinstance(adaptation, dict):
        adaptation = {}
    adaptation["enabled"] = False
    learning["adaptation"] = adaptation
    protection = learning.get("protection")
    if not isinstance(protection, dict):
        protection = {}
    protection["enabled"] = True
    protection["scope"] = identity["scope"]
    identity_section = protection.get("identity")
    if not isinstance(identity_section, dict):
        identity_section = {}
    identity_section["headers"] = headers
    protection["identity"] = identity_section
    tuning = protection.get("tuning")
    if not isinstance(tuning, dict):
        tuning = {}
    tuning["progress_gate"] = {"mode": "observe"}
    protection["tuning"] = tuning
    learning["protection"] = protection

    routing = config.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("template carries no routing section")
    routing["candidate_requirements"] = {"capabilities": "declared"}

    sidecar = {
        "kind": BOOTSTRAP_KIND,
        "client": client,
        "scope": identity["scope"],
        "identity_headers": headers,
        "lanes": [
            {"decision": name, **reason} for name, reason in sorted(reasons.items())
        ],
        "lambda": lam,
        "local_models": [str(m) for m in local_models],
        "adaptation_enabled": False,
        "progress_gate": {"mode": "observe"},
        "candidate_requirements": {"capabilities": "declared"},
        "decision_model": _decision_model_of(config),
    }
    return config, sidecar


def _decision_model_of(config: dict[str, Any]) -> Any:
    """The template's pinned decision model, top-level or canonical layout."""
    for holder in (
        config.get("model_catalog"),
        config.get("global", {}).get("model_catalog"),
    ):
        if isinstance(holder, dict):
            system = holder.get("system")
            if isinstance(system, dict):
                return system.get("decision_model")
    return None


def _decisions_of(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """The template's decisions list and the mapping that owns it.

    The canonical layout carries ``routing.decisions`` (the shipped agent
    recipe); the synthetic test templates carry a top-level ``decisions``.
    The rewrite mutates the owning mapping so either layout round-trips.
    """
    decisions = config.get("decisions")
    if isinstance(decisions, list):
        return decisions, config
    routing = config.get("routing")
    if isinstance(routing, dict):
        decisions = routing.get("decisions")
        if isinstance(decisions, list):
            return decisions, routing
    return None, config


def _pin_local_model(local_models: Sequence[str]) -> str:
    models = [model for model in local_models if str(model).strip()]
    if not models:
        raise ValueError(
            "the privacy/security lanes need a contract-declared local model: "
            "pass local_models (the traffic must never leave the deployment)"
        )
    return str(models[0])
