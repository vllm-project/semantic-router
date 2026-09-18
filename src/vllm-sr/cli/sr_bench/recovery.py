"""Explicit new attempts with immutable lineage and no implicit generation retries."""

from __future__ import annotations

import copy

from .accounting import correction_metadata, effective_calls
from .contracts import digest, planned_cells
from .store import TERMINAL, RecoveryClaimError

MODES = {"undispatched", "failed"}
TERMINAL_FINISH = {"stop", "tool_calls", "function_call", "length"}


class RecoveryPlanError(ValueError):
    """Eligibility changed before any new attempt could dispatch a call."""


def recovery_plan(store, parent_id, mode="undispatched"):
    if mode not in MODES:
        raise ValueError("Recovery mode must be undispatched or failed")
    parent = store.get(parent_id)
    if parent["status"] not in TERMINAL or parent["manifest"]["mode"] != "live":
        raise ValueError("Recovery requires a terminal live parent attempt")
    results = {
        (row["case_id"], row["target_id"]): row for row in store.results(parent_id)
    }
    with store.lock:
        calls = effective_calls(store, parent_id, summary=True)
        accounting = correction_metadata(store, parent_id)
    grouped = {}
    for call in calls:
        grouped.setdefault((call["case_id"], call["target_id"]), []).append(call)
    claimed = store.recovery_claims(parent_id)
    eligible, excluded = [], []
    for cell in planned_cells(parent["manifest"]):
        key = cell["case_id"], cell["target_id"]
        row = results.get(key, {})
        attempts = grouped.get(key, [])
        reason = None
        if row.get("status") == "completed":
            reason = "completed_evidence_is_preserved"
        elif key in claimed:
            reason = "already_claimed_by_child_attempt"
        elif any(call["status"] in {"sent", "sent_unknown"} for call in attempts):
            reason = "ambiguous_dispatch_requires_reconciliation"
        elif row.get("status") == "sent_unknown":
            reason = "ambiguous_saved_result_requires_reconciliation"
        elif mode == "undispatched" and attempts:
            reason = "already_dispatched_requires_explicit_new_attempt"
        elif mode == "failed" and not attempts:
            reason = "never_dispatched_use_undispatched_mode"
        elif mode == "failed" and any(
            call.get("usage") is None
            or call.get("cost_usd") is None
            or call.get("finish_reason") not in TERMINAL_FINISH
            or call["status"] not in {"completed", "failed", "cancelled"}
            for call in attempts
        ):
            reason = "unfinished_or_unknown_accounting_requires_reconciliation"
        if reason:
            excluded.append({**cell, "reason": reason})
        else:
            eligible.append(cell)
    snapshot = {
        "status": parent["status"],
        "progress": parent["progress"],
        "known_spend_usd": sum(call.get("cost_usd") or 0 for call in calls),
        "spend_complete": all(call.get("cost_usd") is not None for call in calls),
        "plan_sha256": parent["manifest"]["plan_sha256"],
    }
    if accounting:
        snapshot["accounting_correction"] = accounting
    proposed = {
        "parent_run_id": parent_id,
        "mode": mode,
        "eligible_cells": eligible,
        "excluded": excluded,
        "counts": {"eligible": len(eligible), "excluded": len(excluded)},
        "parent": snapshot,
        "new_attempt_budget_usd": parent["manifest"]["limits"]["max_cost_usd"],
        "requires_new_attempt_acknowledgment": mode == "failed",
        "scope": "Selected cells only; parent evidence and spend remain separate.",
    }
    return {**proposed, "plan_sha256": digest(proposed)}


def recover(engine, parent_id, body, owner="local"):
    mode = body.get("mode", "undispatched")
    key = body.get("idempotency_key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Recovery requires an explicit idempotency_key")
    chosen = body.get("cells")
    if (
        not isinstance(chosen, list)
        or not chosen
        or any(
            not isinstance(cell, dict)
            or set(cell) != {"case_id", "target_id"}
            or not all(isinstance(value, str) for value in cell.values())
            for cell in chosen
        )
    ):
        raise ValueError("Recovery requires the exact selected case/target cells")
    selected = {(cell["case_id"], cell["target_id"]) for cell in chosen}
    if len(selected) != len(chosen):
        raise ValueError("Duplicate recovery cells are not allowed")
    if mode == "failed" and body.get("acknowledge_new_attempt") is not True:
        raise ValueError(
            "Failed attempts require explicit new-attempt/spend acknowledgment"
        )
    existing = engine.store.request(owner, key)
    if existing:
        prior = existing["manifest"].get("recovery", {})
        if (
            prior.get("parent_run_id") != parent_id
            or prior.get("mode") != mode
            or {(c["case_id"], c["target_id"]) for c in prior.get("selected_cells", [])}
            != selected
        ):
            raise ValueError("Idempotency key is bound to another attempt")
        return existing
    proposed = recovery_plan(engine.store, parent_id, mode)
    if body.get("plan_sha256") != proposed["plan_sha256"]:
        raise RecoveryPlanError(
            "Recovery eligibility changed; inspect a fresh recovery plan"
        )
    available = {(c["case_id"], c["target_id"]) for c in proposed["eligible_cells"]}
    if not selected <= available:
        raise RecoveryPlanError(
            "Selected cells are not eligible for this recovery mode"
        )
    parent = engine.store.get(parent_id)
    manifest = copy.deepcopy(parent["manifest"])
    chosen = [
        cell
        for cell in planned_cells(manifest)
        if (cell["case_id"], cell["target_id"]) in selected
    ]
    target_ids = {target for _, target in selected}
    removed = [
        target for target in manifest["targets"] if target["id"] not in target_ids
    ]
    manifest["targets"] = [
        target for target in manifest["targets"] if target["id"] in target_ids
    ]
    # A baseline subject may remain the frozen judge/simulator of a child target.
    referenced = {
        options[role]
        for options in manifest.get("benchmark_options", {}).values()
        for role in ("judge", "simulator")
        if role in options
    }
    auxiliary = manifest.setdefault("auxiliary_targets", {})
    for target in removed:
        if target["id"] in referenced:
            auxiliary[target["id"]] = target
    manifest["execution_cells"] = chosen
    manifest["name"] = "Recovery subset: " + manifest["name"]
    manifest["recovery"] = {
        "parent_run_id": parent_id,
        "mode": mode,
        "selected_cells": chosen,
        "selected_cells_sha256": digest(chosen),
        "parent_snapshot": proposed["parent"],
        "recovery_subset": True,
        "new_attempt_acknowledged": mode == "failed",
    }
    manifest.pop("plan_sha256", None)
    try:
        return engine.start(manifest, owner, key, recovery=True)
    except RecoveryClaimError as exc:
        raise RecoveryPlanError(
            "Recovery eligibility changed; inspect a fresh recovery plan"
        ) from exc
