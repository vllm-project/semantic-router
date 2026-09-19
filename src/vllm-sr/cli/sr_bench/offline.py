"""No-inference reuse of immutable live answers; never a new formal live score."""

from __future__ import annotations

import copy

from . import VERSION
from .accounting import BUCKETS, correction_metadata, effective_calls
from .contracts import digest
from .engine import basic_grade
from .provenance import capture_runner

BASIC = {"mmlu-pro", "gpqa-diamond", "arc-agi-2"}
REPLAYABLE = BASIC | {"hle", "simpleqa-verified"}


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


def replay(store, baseline_id, preview_id, owner="local", request_key=None):
    baseline, preview = store.get(baseline_id), store.get(preview_id)
    bm, pm = baseline["manifest"], preview["manifest"]
    if (
        baseline["status"] != "completed"
        or bm["mode"] != "live"
        or preview["status"] != "completed"
        or pm["mode"] != "preview"
    ):
        raise ValueError(
            "Replay requires a completed live single-model matrix and a completed preview"
        )
    for key in ("case_sha256", "sampling"):
        if bm[key] != pm[key]:
            raise ValueError(f"Replay requires identical {key}")
    if any(c["benchmark"] not in REPLAYABLE for c in bm["cases"]):
        raise ValueError(
            "Agent and code harness trajectories cannot be replayed as single selections"
        )
    singles = [t for t in bm["targets"] if t["kind"] == "single"]
    if not singles or len({t["model"] for t in singles}) != len(singles):
        raise ValueError("Replay baseline must contain unique single-model identities")
    by_model = {t["model"]: t for t in singles}
    baseline_rows = {
        (r["case_id"], r["target_id"]): r for r in store.results(baseline_id)
    }
    baseline_calls = effective_calls(store, baseline_id)
    preview_rows = {
        (r["case_id"], r["target_id"]): r for r in store.results(preview_id)
    }
    materialized = []
    for target in pm["targets"]:
        for case in pm["cases"]:
            row = preview_rows[(case["id"], target["id"])]
            routing = row.get("details", {}).get("routing", {})
            provenance = routing.get("selection_provenance") or {}
            if (
                provenance.get("state_dependent")
                or provenance.get("mode") == "read_only_snapshot"
            ):
                raise ValueError(
                    "State-dependent learning previews cannot be replayed as stateless single selections"
                )
            decision = routing.get("decision_result")
            if routing.get("selection_status") != "selected" or routing.get(
                "selection_method"
            ) not in {"static", "single", "route_action"}:
                raise ValueError(
                    "Replay requires a deterministic selected single-model route; dynamic or execution-required choices are excluded"
                )
            if not isinstance(decision, dict) or decision.get("plugins"):
                raise ValueError(
                    "Replay excludes routes with request or response plugins"
                )
            selected = by_model.get(routing.get("selected_model"))
            if selected is None:
                raise ValueError(
                    "Preview selected a model absent from the live answer matrix"
                )
            baseline_params = {**bm["sampling"], **selected.get("request_params", {})}
            preview_params = {**pm["sampling"], **target.get("request_params", {})}
            if baseline_params != preview_params:
                raise ValueError(
                    "Replay selected model has different frozen request parameters"
                )
            result = baseline_rows.get((case["id"], selected["id"]))
            calls = [
                c
                for c in baseline_calls
                if c["case_id"] == case["id"]
                and c["target_id"] == selected["id"]
                and c["role"] == "subject"
            ]
            if (
                result is None
                or result["status"] != "completed"
                or len(calls) != 1
                or calls[0]["status"] != "completed"
            ):
                raise ValueError(
                    "Replay requires exactly one saved complete subject generation per selected case"
                )
            if calls[0].get("request", {}).get("messages") != case["messages"]:
                raise ValueError("Saved generated prompt differs from the replay case")
            materialized.append((case, target, result, calls[0], selected["id"]))
    manifest = copy.deepcopy(pm)
    manifest.update(
        {
            "mode": "replay",
            "name": "Offline replay: " + pm["name"],
            "replay_sources": {
                "baseline_run_id": baseline_id,
                "preview_run_id": preview_id,
            },
            "benchmark_options": bm.get("benchmark_options", {}),
            "auxiliary_targets": bm.get("auxiliary_targets", {}),
        }
    )
    manifest["plan_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "plan_sha256"}
    )
    run, created = store.create(
        manifest, owner, request_key, provenance=capture_runner(manifest)
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
