"""No-inference reuse of immutable live answers; never a new formal live score."""

from __future__ import annotations

import copy

from . import VERSION
from .accounting import BUCKETS, correction_metadata, effective_calls
from .contracts import digest
from .engine import basic_grade
from .provenance import capture_runner
from .replay_validation import ReplayEligibilityError, ReplayValidator

BASIC = {"mmlu-pro", "gpqa-diamond", "arc-agi-2"}


def regrade(store, run_id):
    run = store.get(run_id)
    if run["status"] != "completed" or run["manifest"]["mode"] != "live":
        raise ValueError("Regrading requires a completed live run")
    cases = {case["id"]: case for case in run["manifest"]["cases"]}
    if any(case["benchmark"] not in BASIC for case in cases.values()):
        raise ValueError(
            "Offline regrading supports MCQ and grid graders only; judges and harnesses require an explicit new protocol"
        )
    calls = effective_calls(store, run_id)
    results = []
    changed = 0
    for original in store.results(run_id):
        subject = [
            c
            for c in calls
            if c["case_id"] == original["case_id"]
            and c["target_id"] == original["target_id"]
            and c["role"] == "subject"
        ]
        if len(subject) != 1 or subject[0]["status"] != "completed":
            raise ValueError(
                "Offline regrading requires one complete saved subject response per case"
            )
        call = subject[0]
        grade = (
            basic_grade(cases[original["case_id"]], call["final"])
            if call.get("output_complete", True)
            else {
                "answer": None,
                "correct": False,
                "score": 0.0,
                "details": {"quality_failure": "output_limit"},
            }
        )
        changed += grade["correct"] != original["correct"]
        results.append(
            {
                "case_id": original["case_id"],
                "target_id": original["target_id"],
                **grade,
            }
        )
    return {
        "version": VERSION,
        "source_run_id": run_id,
        "kind": "offline-regrade",
        "grader_version": "sr-bench-final-v1",
        "source_plan_sha256": run["manifest"]["plan_sha256"],
        "results": results,
        "changed_count": changed,
        "model_requests": 0,
    }


def export_training(store, run_id):
    run = store.get(run_id)
    if run["status"] != "completed" or run["manifest"]["mode"] != "live":
        raise ValueError("Training export requires a completed live run")
    manifest = run["manifest"]
    if manifest.get("recovery") or manifest.get("execution_cells"):
        raise ValueError(
            "Training export requires a complete rectangular matrix; recovery subsets stay separate"
        )
    if manifest["profile"] == "standard" or any(
        c.get("metadata", {}).get("split") != "dev" for c in manifest["cases"]
    ):
        raise ValueError(
            "Training export accepts only explicitly marked dev cases; holdout and unknown splits are excluded"
        )
    results = {(r["case_id"], r["target_id"]): r for r in store.results(run_id)}
    calls = effective_calls(store, run_id)
    rows = []
    for case in manifest["cases"]:
        entries = []
        for target in manifest["targets"]:
            row = results[(case["id"], target["id"])]
            subject = [
                c
                for c in calls
                if c["case_id"] == case["id"]
                and c["target_id"] == target["id"]
                and c["role"] == "subject"
            ]
            entries.append(
                {
                    "id": target["id"],
                    "model": target["model"],
                    "kind": target["kind"],
                    "request_params": target.get("request_params", {}),
                    **{
                        k: row.get(k)
                        for k in (
                            "answer",
                            "correct",
                            "score",
                            "usage",
                            "cost_usd",
                            "subject_latency_s",
                        )
                    },
                    "usage": (
                        {k: sum(c["usage"][k] for c in subject) for k in BUCKETS}
                        if subject and all(c.get("usage") is not None for c in subject)
                        else None
                    ),
                    "cost_usd": (
                        sum(c["cost_usd"] for c in subject)
                        if subject
                        and all(c.get("cost_usd") is not None for c in subject)
                        else None
                    ),
                    "final": subject[-1].get("final") if subject else None,
                }
            )
        rows.append({**case, "targets": entries})
    return {
        "version": VERSION,
        "source_run_id": run_id,
        "kind": "training-matrix",
        "split": "dev",
        "cases": rows,
        "source_plan_sha256": manifest["plan_sha256"],
        "accounting_correction": correction_metadata(store, run_id),
        "model_requests": 0,
    }


def replay(
    store,
    baseline_id,
    preview_id,
    owner="local",
    request_key=None,
    *,
    actor_role="local",
):
    # Serialize the key lookup, validation and receipt creation. A known rejected
    # request cannot race a successful replay with the same idempotency key.
    with store.lock:
        if request_key and (existing := store.request(owner, request_key)):
            sources = existing["manifest"].get("replay_sources")
            if sources != {
                "baseline_run_id": baseline_id,
                "preview_run_id": preview_id,
            }:
                raise ValueError(
                    "idempotency key is already bound to a different replay"
                )
            return existing
        return _materialize_replay(
            store, baseline_id, preview_id, owner, request_key, actor_role
        )


def _materialize_replay(store, baseline_id, preview_id, owner, request_key, actor_role):
    baseline, preview = store.get(baseline_id), store.get(preview_id)
    bm, pm = baseline["manifest"], preview["manifest"]
    validation = ReplayValidator(store, baseline).validate(preview)
    if not validation["eligible"]:
        raise ReplayEligibilityError(validation["reasons"])
    materialized = validation["materialized"]
    manifest = copy.deepcopy(pm)
    manifest.update(
        {
            "mode": "replay",
            "name": "Offline replay: " + pm["name"],
            "replay_sources": {
                "baseline_run_id": baseline_id,
                "preview_run_id": preview_id,
            },
            "replay_compatibility": validation["receipt"],
            "benchmark_options": bm.get("benchmark_options", {}),
            "auxiliary_targets": bm.get("auxiliary_targets", {}),
        }
    )
    if "experiment" in manifest:
        manifest["experiment"]["role"] = "estimate"
    singles = {t["id"]: t for t in bm["targets"] if t["kind"] == "single"}
    for target in manifest["targets"]:
        target["prices"] = {
            model: price
            for _, selected_target, _, _, source_target in materialized
            if selected_target["id"] == target["id"]
            for model, price in singles[source_target].get("prices", {}).items()
        }
    manifest["plan_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "plan_sha256"}
    )
    run, created = store.create(
        manifest,
        owner,
        request_key,
        provenance=capture_runner(manifest),
        actor_role=actor_role,
    )
    if not created:
        return run
    run_id = run["id"]
    for case, target, original, call, source_target in materialized:
        result = {
            **original,
            "usage": call.get("usage"),
            "cost_usd": call.get("cost_usd"),
            "details": {
                **original.get("details", {}),
                "evidence_mode": "offline_estimate",
                "source_run_id": baseline_id,
                "source_target_id": source_target,
            },
        }
        store.result(run_id, case["id"], target["id"], "completed", result)
        data = {
            k: v
            for k, v in call.items()
            if k not in {"id", "case_id", "target_id", "role", "status"}
        }
        store.cached_call(
            run_id,
            case["id"],
            target["id"],
            {**data, "source_call_id": call["id"], "source_run_id": baseline_id},
        )
    store.status(run_id, "completed")
    return store.get(run_id)
