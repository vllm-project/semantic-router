"""Derive a candidate plan from a saved baseline without resampling questions."""

import copy


def candidate_manifest(baseline, targets, mode="live", name=None, experiment=None):
    source = baseline["manifest"]
    if (
        baseline["status"] != "completed"
        or source["mode"] != "live"
        or not any(target["kind"] == "single" for target in source["targets"])
    ):
        raise ValueError("Choose a completed live single-model baseline")
    if source.get("execution_cells") is not None:
        raise ValueError("A recovery subset is not a complete reusable baseline")
    if not targets or any(target.get("kind") != "mom" for target in targets):
        raise ValueError("Candidate plans require configured MoM targets")
    if mode not in {"live", "preview"}:
        raise ValueError("Candidate mode must be live or preview")
    manifest = {
        key: copy.deepcopy(source[key])
        for key in (
            "version",
            "profile",
            "seed",
            "cases",
            "dataset",
            "sampling",
            "limits",
            "benchmark_options",
            "auxiliary_targets",
            "adapter_versions",
            "benchmark_weights",
            "cost_policy",
        )
        if key in source
    }
    manifest.update(
        {
            "name": name
            or ("Routing check" if mode == "preview" else "Candidate evaluation"),
            "mode": mode,
            "targets": copy.deepcopy(targets),
            "baseline_run_id": baseline["id"],
        }
    )
    if experiment is not None:
        manifest["experiment"] = experiment
    return manifest


def validate_candidate_protocol(baseline, candidate):
    """Server configuration must not silently change a reusable baseline protocol."""
    source = baseline["manifest"]
    for key in (
        "case_sha256",
        "sampling",
        "limits",
        "profile",
        "seed",
        "cost_policy",
        "benchmark_options",
        "auxiliary_targets",
        "adapter_versions",
        "benchmark_weights",
    ):
        default = {} if key in {"benchmark_options", "auxiliary_targets"} else None
        if candidate.get(key, default) != source.get(key, default):
            raise ValueError(
                f"Baseline protocol changed ({key}); prepare a new baseline before comparing this configuration"
            )
