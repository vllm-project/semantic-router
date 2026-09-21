"""Shared selection provenance validation for writers, readers and frozen plans."""

from __future__ import annotations

from collections import Counter

from .canonical import digest
from .dataset_io import MAX_ROWS
from .history_snapshot import validate_snapshot
from .task_identity import SHA256, source_identity, task_key

SELECTION = "stratified-hash-provenance-v1"
SCHEMA = "sr-bench-preparation-v1"
FIELDS = {
    "schema",
    "evaluation_role",
    "task_source",
    "history_snapshot",
    "quick_task_keys",
    "source_count",
    "requested_count",
    "eligible_count",
    "selected_count",
    "excluded_count",
    "selected_task_keys_sha256",
    "selected_case_ids_sha256",
    "coverage",
}


def summary(manifest):
    """Compact collection provenance; full proofs stay in the frozen manifest."""
    return {
        family: {
            **{
                key: entry[key]
                for key in (
                    "evaluation_role",
                    "coverage",
                    "source_count",
                    "requested_count",
                    "eligible_count",
                    "selected_count",
                    "excluded_count",
                )
            },
            "history_snapshot_id": (
                entry["history_snapshot"]["id"] if entry["history_snapshot"] else None
            ),
        }
        for family, entry in manifest.get("preparation", {}).items()
    }


def dataset_identity(manifest):
    content = {
        "cases_sha256": manifest["sha256"],
        "sources": manifest["sources"],
        "profile": manifest["profile"],
        "seed": manifest["seed"],
        "custom_subset": manifest.get("custom_subset", False),
        "selection": manifest["selection"],
    }
    if "preparation" in manifest:
        content["preparation"] = manifest["preparation"]
    return digest(content)


def select_cases(
    ordered,
    benchmark,
    quick_count,
    desired,
    *,
    snapshot=None,
    role=None,
    task_source=None,
):
    if role not in (None, "holdout", "retest"):
        raise ValueError("Evaluation role must be holdout or retest")
    if snapshot is not None:
        validate_snapshot(snapshot)
        role = role or "holdout"
        if task_source is None:
            raise ValueError("History exclusion requires an explicit source partition")
    if role == "holdout" and snapshot is None:
        raise ValueError("A holdout role requires an explicit history snapshot")
    keys = [task_key(case, task_source) for case in ordered] if task_source else []
    if keys and len(set(keys)) != len(keys):
        raise ValueError("Duplicate canonical source task identities")
    quick = set(keys[:quick_count])
    history = set()
    if snapshot is not None and benchmark in snapshot["families"]:
        family = snapshot["families"][benchmark]
        if family["source"] != task_source:
            raise ValueError(
                "Cross-source or cross-revision history requires explicit reconciliation"
            )
        history = set(family["task_keys"])
        if not history <= set(keys):
            raise ValueError("History task identities are absent from the exact source")
    excluded = quick | history
    eligible = (
        [case for case, key in zip(ordered, keys, strict=True) if key not in excluded]
        if task_source
        else ordered[quick_count:]
    )
    if len(eligible) < desired:
        raise ValueError(
            f"Insufficient tasks after frozen exclusions: requested={desired}, "
            f"eligible={len(eligible)}, quick={quick_count}, history={len(history)}, "
            f"quick_history_overlap={len(quick & history)}"
        )
    selected = eligible[:desired]
    entry = {
        "schema": SCHEMA,
        "evaluation_role": role,
        "task_source": task_source,
        "history_snapshot": snapshot,
        "quick_task_keys": sorted(quick),
        "source_count": len(ordered),
        "requested_count": desired,
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "excluded_count": len(ordered) - len(eligible),
        "selected_task_keys_sha256": (
            digest(sorted(task_key(case, task_source) for case in selected))
            if task_source
            else None
        ),
        "selected_case_ids_sha256": digest(sorted(case["id"] for case in selected)),
        "coverage": (
            "named-memberships-only" if snapshot else "no-history-qualification"
        ),
    }
    return selected, {benchmark: entry}


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be an object")
    preparation = manifest.get("preparation")
    if preparation is None:
        if manifest.get("selection") == SELECTION:
            raise ValueError("Prepared selection provenance is missing")
        return
    sources = manifest.get("sources")
    if (
        not isinstance(sources, dict)
        or not isinstance(preparation, dict)
        or not preparation
        or not set(preparation) <= set(sources)
    ):
        raise ValueError("Invalid per-family preparation provenance")
    if manifest.get("benchmarks") != sorted(sources):
        raise ValueError("Prepared benchmark scope differs from source provenance")
    if (
        type(manifest.get("case_count")) is not int
        or not 1 <= manifest["case_count"] <= MAX_ROWS
    ):
        raise ValueError("Invalid prepared dataset count")
    if manifest.get("split") != (
        "holdout" if manifest.get("profile") == "standard" else "dev"
    ):
        raise ValueError("Prepared evaluation split differs from profile")
    if manifest.get("selection") != SELECTION or dataset_identity(
        manifest
    ) != manifest.get("id"):
        raise ValueError("Prepared dataset provenance identity changed")
    for family, entry in preparation.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != FIELDS
            or entry["schema"] != SCHEMA
        ):
            raise ValueError("Invalid frozen preparation fields")
        if entry["evaluation_role"] not in (None, "holdout", "retest"):
            raise ValueError("Invalid frozen evaluation role")
        for name in (
            "source_count",
            "requested_count",
            "eligible_count",
            "selected_count",
            "excluded_count",
        ):
            if type(entry[name]) is not int or entry[name] < 0:
                raise ValueError("Preparation counts must be nonnegative integers")
        if (
            not 0
            < entry["selected_count"]
            == entry["requested_count"]
            <= entry["eligible_count"]
        ):
            raise ValueError(
                "Prepared selection does not meet its exact requested count"
            )
        if entry["eligible_count"] + entry["excluded_count"] != entry["source_count"]:
            raise ValueError("Prepared eligibility counts are inconsistent")
        task_source = entry["task_source"]
        quick = entry["quick_task_keys"]
        if (
            not isinstance(quick, list)
            or any(
                not isinstance(key, str) or not SHA256.fullmatch(key) for key in quick
            )
            or quick != sorted(set(quick))
        ):
            raise ValueError("Invalid frozen Quick task membership")
        if not isinstance(
            entry["selected_case_ids_sha256"], str
        ) or not SHA256.fullmatch(entry["selected_case_ids_sha256"]):
            raise ValueError("Invalid selected case identity digest")
        if task_source is not None:
            if (
                not isinstance(task_source, dict)
                or source_identity(
                    manifest["sources"][family], task_source.get("partition")
                )
                != task_source
            ):
                raise ValueError("Preparation source identity changed")
            if not isinstance(
                entry["selected_task_keys_sha256"], str
            ) or not SHA256.fullmatch(entry["selected_task_keys_sha256"]):
                raise ValueError("Invalid selected task identity digest")
        snapshot = entry["history_snapshot"]
        if snapshot is not None:
            validate_snapshot(snapshot)
            if task_source is None or entry["coverage"] != "named-memberships-only":
                raise ValueError("History qualification lacks source identity")
            if entry["evaluation_role"] not in {"holdout", "retest"}:
                raise ValueError("History-qualified preparation requires a frozen role")
            if (
                family in snapshot["families"]
                and snapshot["families"][family]["source"] != task_source
            ):
                raise ValueError("History snapshot uses a different source")
        elif (
            entry["evaluation_role"] == "holdout"
            or entry["coverage"] != "no-history-qualification"
        ):
            raise ValueError("Unsupported disjoint holdout claim without history")
        if task_source is None and (
            entry["quick_task_keys"] or entry["selected_task_keys_sha256"] is not None
        ):
            raise ValueError("Task identity proof requires a frozen source")
        if task_source is not None:
            reserved = set(quick)
            if snapshot is not None:
                reserved.update(
                    snapshot["families"].get(family, {}).get("task_keys", [])
                )
            if len(reserved) != entry["excluded_count"]:
                raise ValueError("Frozen exclusion union differs from its count")


def validate_cases(manifest, cases):
    validate_manifest(manifest)
    preparation = manifest.get("preparation", {})
    counts = Counter()
    keys, ids, reserved = {}, {}, {}
    for family, entry in preparation.items():
        reserved[family] = set(entry["quick_task_keys"])
        history = entry["history_snapshot"]
        if history is not None:
            reserved[family].update(
                history["families"].get(family, {}).get("task_keys", [])
            )
    for case in cases:
        family = case["benchmark"]
        counts[family] += 1
        if family not in preparation:
            continue
        entry = preparation[family]
        ids.setdefault(family, []).append(case["id"])
        if entry["task_source"] is not None:
            key = task_key(case, entry["task_source"])
            if key in keys.setdefault(family, set()):
                raise ValueError("Duplicate canonical prepared task identity")
            keys[family].add(key)
            if key in reserved[family]:
                raise ValueError("Prepared task overlaps its frozen exclusions")
    for family, entry in preparation.items():
        if (
            counts[family] != entry["selected_count"]
            or digest(sorted(ids.get(family, []))) != entry["selected_case_ids_sha256"]
        ):
            raise ValueError(
                "Prepared case membership differs from selection provenance"
            )
        if (
            entry["task_source"] is not None
            and digest(sorted(keys.get(family, set())))
            != entry["selected_task_keys_sha256"]
        ):
            raise ValueError("Prepared canonical task membership changed")
    if preparation and (
        sum(counts.values()) != manifest["case_count"]
        or set(counts) != set(manifest["benchmarks"])
    ):
        raise ValueError("Prepared dataset membership differs from its scope")
