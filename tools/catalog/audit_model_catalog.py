#!/usr/bin/env python3
"""Report authored model-catalog evaluation completeness without mutating it."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CATALOG_TOOL_ROOT = Path(__file__).resolve().parent
if str(CATALOG_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(CATALOG_TOOL_ROOT))

from generate_model_catalog import (  # noqa: E402
    CatalogBuildError,
    load_and_validate,
)

DEFAULT_MIN_EVALUATIONS = 5


def _selected_values(values: Iterable[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or ():
        selected.update(part.strip() for part in value.split(",") if part.strip())
    return selected


def _evaluation_efforts(
    model: dict[str, Any],
    reasoning_families: dict[str, dict[str, Any]],
    observed_efforts: set[str],
) -> list[str]:
    family_id = model.get("reasoning_family")
    if family_id is None:
        return sorted(observed_efforts) if observed_efforts else ["default"]
    declared = list(reasoning_families[str(family_id)]["levels"])
    return declared + sorted(observed_efforts.difference(declared))


def _declared_efforts(
    model: dict[str, Any],
    reasoning_families: dict[str, dict[str, Any]],
) -> list[str]:
    family_id = model.get("reasoning_family")
    if family_id is None:
        return ["default"]
    return list(reasoning_families[str(family_id)]["levels"])


def _default_slots(
    manifest: dict[str, Any], resources: dict[str, list[dict[str, Any]]]
) -> list[dict[str, str]]:
    index_id = str(manifest["defaults"]["intelligence_index"])
    definition = next(item for item in resources["indices"] if item["id"] == index_id)
    return [
        {
            "benchmark": str(component["benchmark"]),
            "benchmark_profile": str(component["benchmark_profile"]),
            "metric": str(component["metric"]),
        }
        for component in definition["components"]
        if component.get("benchmark")
    ]


def _matches_filters(
    model: dict[str, Any], publishers: set[str], model_ids: set[str]
) -> bool:
    if publishers and str(model.get("publisher", "")).casefold() not in publishers:
        return False
    return not model_ids or str(model["id"]) in model_ids


def _select_audit_inputs(
    resources: dict[str, list[dict[str, Any]]],
    publishers: Iterable[str] | None,
    models: Iterable[str] | None,
) -> tuple[
    set[str],
    set[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    publisher_filters = set(publishers or ())
    requested_publishers = {value.casefold() for value in publisher_filters}
    requested_models = set(models or ())
    known_publishers = {
        str(model.get("publisher", "")) for model in resources["models"]
    }
    known_publishers_folded = {item.casefold() for item in known_publishers}
    unknown_publishers = sorted(
        value
        for value in publisher_filters
        if value.casefold() not in known_publishers_folded
    )
    known_model_ids = {str(model["id"]) for model in resources["models"]}
    unknown_models = sorted(requested_models.difference(known_model_ids))
    if unknown_publishers:
        raise CatalogBuildError(
            "unknown catalog publisher filter: " + ", ".join(unknown_publishers)
        )
    if unknown_models:
        raise CatalogBuildError(
            "unknown catalog model filter: " + ", ".join(unknown_models)
        )

    selected_models = sorted(
        (
            model
            for model in resources["models"]
            if _matches_filters(model, requested_publishers, requested_models)
        ),
        key=lambda item: str(item["id"]),
    )
    if (requested_publishers or requested_models) and not selected_models:
        raise CatalogBuildError("catalog audit filters select no models")
    selected_ids = {str(model["id"]) for model in selected_models}
    selected_evaluations = [
        item for item in resources["evaluations"] if str(item["model"]) in selected_ids
    ]
    available = [
        item
        for item in selected_evaluations
        if item.get("status") == "available" and item.get("metrics")
    ]
    return (
        publisher_filters,
        requested_models,
        selected_models,
        selected_evaluations,
        available,
    )


def _model_effort_rows(
    manifest: dict[str, Any],
    resources: dict[str, list[dict[str, Any]]],
    selected_models: list[dict[str, Any]],
    selected_evaluations: list[dict[str, Any]],
    available: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    observed_efforts: dict[str, set[str]] = defaultdict(set)
    available_by_effort: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in selected_evaluations:
        observed_efforts[str(item["model"])].add(str(item["reasoning_effort"]))
    for item in available:
        key = (str(item["model"]), str(item["reasoning_effort"]))
        available_by_effort[key].append(item)

    slots = _default_slots(manifest, resources)
    slot_keys = {
        (slot["benchmark"], slot["benchmark_profile"], slot["metric"]): slot
        for slot in slots
    }
    reasoning_families = {
        str(item["id"]): item for item in resources["reasoning_families"]
    }
    rows: list[dict[str, Any]] = []
    for model in selected_models:
        model_id = str(model["id"])
        family_id = model.get("reasoning_family")
        declared_efforts = (
            set(reasoning_families[str(family_id)]["levels"])
            if family_id is not None
            else set()
        )
        efforts = _evaluation_efforts(
            model, reasoning_families, observed_efforts.get(model_id, set())
        )
        for effort in efforts:
            evaluations = available_by_effort.get((model_id, effort), [])
            benchmarks = sorted({str(item["benchmark"]) for item in evaluations})
            present_slot_keys = {
                (str(item["benchmark"]), str(item["benchmark_profile"]), str(metric))
                for item in evaluations
                for metric in item["metrics"]
            }.intersection(slot_keys)
            available_slots = [
                slot for key, slot in slot_keys.items() if key in present_slot_keys
            ]
            missing_slots = [
                slot for key, slot in slot_keys.items() if key not in present_slot_keys
            ]
            rows.append(
                {
                    "model": model_id,
                    "publisher": str(model.get("publisher", "")),
                    "kind": str(model["kind"]),
                    "reasoning_effort": effort,
                    "selectable": effort in declared_efforts,
                    "available_evaluations": len(evaluations),
                    "available_benchmark_count": len(benchmarks),
                    "available_benchmarks": benchmarks,
                    "default_slots_available_count": len(available_slots),
                    "default_slots_available": available_slots,
                    "default_slots_missing": missing_slots,
                }
            )
    return slots, reasoning_families, rows


def _selectable_effort_coverage(
    model_efforts: list[dict[str, Any]],
) -> dict[str, int]:
    selectable = [
        row for row in model_efforts if row["kind"] == "physical" and row["selectable"]
    ]
    complete = sum(
        int(row["available_benchmark_count"]) >= DEFAULT_MIN_EVALUATIONS
        for row in selectable
    )
    partial = sum(
        0 < int(row["available_benchmark_count"]) < DEFAULT_MIN_EVALUATIONS
        for row in selectable
    )
    unmeasured = sum(int(row["available_benchmark_count"]) == 0 for row in selectable)
    return {
        "minimum_available_benchmarks": DEFAULT_MIN_EVALUATIONS,
        "total": len(selectable),
        "complete": complete,
        "partial": partial,
        "unmeasured": unmeasured,
    }


def _model_evidence_bucket_rows(
    available: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations_by_bucket: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for item in available:
        evidence = item["evidence"]
        key = (
            str(item["model"]),
            str(item["reasoning_effort"]),
            str(evidence["provenance"]),
        )
        evaluations_by_bucket[key].append(item)

    rows: list[dict[str, Any]] = []
    for (model, effort, provenance), evaluations in sorted(
        evaluations_by_bucket.items()
    ):
        benchmarks = sorted({str(item["benchmark"]) for item in evaluations})
        rows.append(
            {
                "model": model,
                "reasoning_effort": effort,
                "provenance": provenance,
                "available_evaluations": len(evaluations),
                "available_benchmark_count": len(benchmarks),
                "available_benchmarks": benchmarks,
            }
        )
    return rows


def _model_coverage_rows(
    selected_models: list[dict[str, Any]],
    reasoning_families: dict[str, dict[str, Any]],
    model_efforts: list[dict[str, Any]],
    evidence_buckets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    effort_rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_efforts:
        effort_rows_by_model[str(row["model"])].append(row)
    evidence_buckets_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_buckets:
        evidence_buckets_by_model[str(row["model"])].append(row)

    rows: list[dict[str, Any]] = []
    for model in selected_models:
        model_id = str(model["id"])
        effort_rows = effort_rows_by_model[model_id]
        declared_efforts = _declared_efforts(model, reasoning_families)
        declared_set = set(declared_efforts)
        model_evidence_buckets = evidence_buckets_by_model[model_id]
        best = (
            sorted(
                model_evidence_buckets,
                key=lambda row: (
                    -int(row["available_benchmark_count"]),
                    -int(row["available_evaluations"]),
                    str(row["reasoning_effort"]),
                    str(row["provenance"]),
                ),
            )[0]
            if model_evidence_buckets
            else None
        )
        all_benchmarks = sorted(
            {
                str(benchmark)
                for row in effort_rows
                for benchmark in row["available_benchmarks"]
            }
        )
        evaluated_efforts = [
            str(row["reasoning_effort"])
            for row in effort_rows
            if row["available_evaluations"]
        ]
        rows.append(
            {
                "model": model_id,
                "publisher": str(model.get("publisher", "")),
                "kind": str(model["kind"]),
                "available_evaluations": sum(
                    int(row["available_evaluations"]) for row in effort_rows
                ),
                "available_benchmark_count": len(all_benchmarks),
                "available_benchmarks": all_benchmarks,
                "best_evidence_bucket": (
                    {
                        "reasoning_effort": str(best["reasoning_effort"]),
                        "provenance": str(best["provenance"]),
                    }
                    if best is not None
                    else None
                ),
                "best_evidence_bucket_benchmark_count": (
                    int(best["available_benchmark_count"]) if best is not None else 0
                ),
                "declared_reasoning_efforts": declared_efforts,
                "evaluated_reasoning_efforts": evaluated_efforts,
                "declared_reasoning_efforts_evaluated": [
                    effort for effort in evaluated_efforts if effort in declared_set
                ],
                "evidence_only_efforts": [
                    effort for effort in evaluated_efforts if effort not in declared_set
                ],
            }
        )
    return rows


def _publisher_rows(
    selected_models: list[dict[str, Any]],
    selected_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    publishers = sorted({str(model.get("publisher", "")) for model in selected_models})
    for publisher in publishers:
        publisher_models = [
            model for model in selected_models if model.get("publisher") == publisher
        ]
        publisher_ids = {str(model["id"]) for model in publisher_models}
        rows.append(
            {
                "publisher": publisher,
                "physical_models": sum(
                    model["kind"] == "physical" for model in publisher_models
                ),
                "virtual_models": sum(
                    model["kind"] == "virtual" for model in publisher_models
                ),
                "evaluations": sum(
                    str(item["model"]) in publisher_ids for item in selected_evaluations
                ),
            }
        )
    return rows


def _provider_rows(
    resources: dict[str, list[dict[str, Any]]],
    selected_ids: set[str],
    *,
    filtered: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in sorted(resources["providers"], key=lambda item: str(item["id"])):
        bindings = provider.get("models", [])
        catalog_bindings = [
            item for item in bindings if item.get("catalog") in selected_ids
        ]
        if filtered and not catalog_bindings:
            continue
        rows.append(
            {
                "provider": str(provider["id"]),
                "category": str(provider["category"]),
                "support_tier": str(provider["support_tier"]),
                "model_bindings": len(bindings),
                "catalog_model_bindings": len(catalog_bindings),
                "custom_model_bindings": sum(
                    not item.get("catalog") for item in bindings
                ),
            }
        )
    return rows


def build_audit(
    manifest: dict[str, Any],
    resources: dict[str, list[dict[str, Any]]],
    *,
    publishers: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic audit report from validated authored resources."""

    (
        publisher_filters,
        requested_models,
        selected_models,
        selected_evaluations,
        available,
    ) = _select_audit_inputs(resources, publishers, models)
    selected_ids = {str(model["id"]) for model in selected_models}
    slots, reasoning_families, model_efforts = _model_effort_rows(
        manifest, resources, selected_models, selected_evaluations, available
    )
    model_evidence_buckets = _model_evidence_bucket_rows(available)
    model_coverage = _model_coverage_rows(
        selected_models, reasoning_families, model_efforts, model_evidence_buckets
    )
    publisher_rows = _publisher_rows(selected_models, selected_evaluations)
    provider_rows = _provider_rows(
        resources,
        selected_ids,
        filtered=bool(publisher_filters or requested_models),
    )

    models_without_evaluations = [
        row["model"] for row in model_coverage if not row["available_evaluations"]
    ]
    physical_models_without_evaluations = [
        row["model"]
        for row in model_coverage
        if row["kind"] == "physical" and not row["available_evaluations"]
    ]
    physical_models_below_five = [
        {
            "model": row["model"],
            "best_evidence_bucket": row["best_evidence_bucket"],
            "available_benchmark_count": row["best_evidence_bucket_benchmark_count"],
        }
        for row in model_coverage
        if row["kind"] == "physical"
        and row["best_evidence_bucket_benchmark_count"] < DEFAULT_MIN_EVALUATIONS
    ]
    model_efforts_below_five = [
        {
            "model": row["model"],
            "reasoning_effort": row["reasoning_effort"],
            "available_benchmark_count": row["available_benchmark_count"],
        }
        for row in model_efforts
        if row["kind"] == "physical"
        and row["available_benchmark_count"] < DEFAULT_MIN_EVALUATIONS
    ]
    return {
        "scope": {
            "publishers": sorted(publisher_filters),
            "models": sorted(requested_models),
        },
        "counts": {
            "physical_models": sum(
                model["kind"] == "physical" for model in selected_models
            ),
            "virtual_models": sum(
                model["kind"] == "virtual" for model in selected_models
            ),
            "providers": len(provider_rows),
            "evaluations": len(selected_evaluations),
            "available_evaluations": len(available),
        },
        "publishers": publisher_rows,
        "providers": provider_rows,
        "default_index": str(manifest["defaults"]["intelligence_index"]),
        "default_slots": slots,
        "selectable_effort_coverage": _selectable_effort_coverage(model_efforts),
        "models_without_evaluations": models_without_evaluations,
        "physical_models_without_evaluations": physical_models_without_evaluations,
        "physical_models_below_five": physical_models_below_five,
        "model_efforts_below_five": model_efforts_below_five,
        "model_coverage": model_coverage,
        "model_evidence_buckets": model_evidence_buckets,
        "model_efforts": model_efforts,
    }


def gate_failures(
    report: dict[str, Any], minimum: int, *, scope: str
) -> list[dict[str, Any]]:
    """Return physical model or model-effort rows below an explicit gate."""

    if scope == "model":
        return [
            row
            for row in report["model_coverage"]
            if row["kind"] == "physical"
            and row["best_evidence_bucket_benchmark_count"] < minimum
        ]
    if scope == "effort":
        return [
            row
            for row in report["model_efforts"]
            if row["kind"] == "physical" and row["available_benchmark_count"] < minimum
        ]
    if scope == "selectable-effort":
        return [
            row
            for row in report["model_efforts"]
            if row["kind"] == "physical"
            and row["selectable"]
            and row["available_benchmark_count"] < minimum
        ]
    raise ValueError(f"unsupported gate scope: {scope}")


def _format_slot(slot: dict[str, str]) -> str:
    return f"{slot['benchmark']}[{slot['benchmark_profile']}]#{slot['metric']}"


def _format_evidence_bucket(bucket: dict[str, str] | None) -> str:
    if bucket is None:
        return "none"
    return f"{bucket['reasoning_effort']}/{bucket['provenance']}"


def render_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "Model catalog completeness audit",
        (
            "Counts: "
            f"physical={counts['physical_models']} virtual={counts['virtual_models']} "
            f"providers={counts['providers']} evaluations={counts['evaluations']} "
            f"available={counts['available_evaluations']}"
        ),
        "",
        "Publishers:",
    ]
    lines.extend(
        "  "
        f"{row['publisher']}: physical={row['physical_models']} "
        f"virtual={row['virtual_models']} evaluations={row['evaluations']}"
        for row in report["publishers"]
    )
    lines.extend(["", "Providers:"])
    lines.extend(
        "  "
        f"{row['provider']}: bindings={row['model_bindings']} "
        f"catalog={row['catalog_model_bindings']} custom={row['custom_model_bindings']}"
        for row in report["providers"]
    )
    lines.extend(
        [
            "",
            "Selectable reasoning-effort coverage "
            f"(minimum {report['selectable_effort_coverage']['minimum_available_benchmarks']} "
            "benchmarks): "
            f"complete={report['selectable_effort_coverage']['complete']} "
            f"partial={report['selectable_effort_coverage']['partial']} "
            f"unmeasured={report['selectable_effort_coverage']['unmeasured']}",
            "",
            f"Default slots ({report['default_index']}):",
            *[f"  {_format_slot(slot)}" for slot in report["default_slots"]],
            "",
            "Physical model evidence coverage:",
        ]
    )
    for row in report["model_coverage"]:
        if row["kind"] != "physical":
            continue
        lines.append(
            f"  {row['model']}: "
            f"best={_format_evidence_bucket(row['best_evidence_bucket'])} "
            f"benchmarks={row['best_evidence_bucket_benchmark_count']} "
            "declared_efforts="
            f"{len(row['declared_reasoning_efforts_evaluated'])}/"
            f"{len(row['declared_reasoning_efforts'])} "
            f"evidence_only_efforts={len(row['evidence_only_efforts'])}"
        )
    lines.extend(["", "Model x reasoning effort:"])
    for row in report["model_efforts"]:
        missing = ", ".join(_format_slot(slot) for slot in row["default_slots_missing"])
        lines.append(
            f"  {row['model']} [{row['reasoning_effort']}]: "
            f"benchmarks={row['available_benchmark_count']} "
            f"default_slots={row['default_slots_available_count']}/"
            f"{len(report['default_slots'])}"
            + (f" missing={missing}" if missing else "")
        )
    lines.extend(
        [
            "",
            "Models without available evaluations: "
            + (
                ", ".join(report["models_without_evaluations"])
                if report["models_without_evaluations"]
                else "none"
            ),
        ]
    )
    lines.extend(
        ["", "Physical models below five benchmarks in any exact evidence bucket:"]
    )
    lines.extend(
        f"  {row['model']} "
        f"[{_format_evidence_bucket(row['best_evidence_bucket'])}]: "
        f"benchmarks={row['available_benchmark_count']}"
        for row in report["physical_models_below_five"]
    )
    if not report["physical_models_below_five"]:
        lines.append("  none")
    lines.extend(["", "Model-effort rows below five available benchmarks:"])
    lines.extend(
        f"  {row['model']} [{row['reasoning_effort']}]: "
        f"benchmarks={row['available_benchmark_count']}"
        for row in report["model_efforts_below_five"]
    )
    if not report["model_efforts_below_five"]:
        lines.append("  none")
    return "\n".join(lines) + "\n"


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publishers",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="audit only models from these publishers (repeatable)",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        metavar="ID[,ID...]",
        help="audit only these exact catalog model IDs (repeatable)",
    )
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--require-min-evaluations-per-model",
        type=_nonnegative_int,
        metavar="N",
        help=(
            "fail when a selected physical model has fewer than N benchmarks "
            "in every exact reasoning-effort/provenance evidence bucket"
        ),
    )
    gate_group.add_argument(
        "--require-min-evaluations-per-effort",
        type=_nonnegative_int,
        metavar="N",
        help=(
            "fail when any declared or observed physical model-effort has fewer "
            "than N benchmarks"
        ),
    )
    gate_group.add_argument(
        "--require-min-evaluations-per-selectable-effort",
        type=_nonnegative_int,
        metavar="N",
        help=(
            "fail when a declared reasoning-family level has fewer than N "
            "available benchmarks; evidence-only and unspecified buckets are ignored"
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args(argv)
    publishers = _selected_values(args.publishers)
    models = _selected_values(args.models)
    try:
        manifest, resources, _ = load_and_validate()
        report = build_audit(manifest, resources, publishers=publishers, models=models)
    except CatalogBuildError as error:
        print(f"model catalog audit failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(render_text(report), end="")
    if args.require_min_evaluations_per_model is not None:
        minimum = args.require_min_evaluations_per_model
        scope = "model"
    elif args.require_min_evaluations_per_effort is not None:
        minimum = args.require_min_evaluations_per_effort
        scope = "effort"
    elif args.require_min_evaluations_per_selectable_effort is not None:
        minimum = args.require_min_evaluations_per_selectable_effort
        scope = "selectable-effort"
    else:
        return 0
    failures = gate_failures(report, minimum, scope=scope)
    if failures:
        print(
            f"model catalog audit gate failed: {len(failures)} physical {scope} rows "
            f"have fewer than {minimum} available benchmarks",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
