from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from cli.sr_bench.contracts import DEFAULT_LIMITS, VERSION, load_document
from cli.sr_bench.dataset_io import MAX_SCAN_BYTES, read_small
from cli.sr_bench.evalscope_gpqa_worker import prepare_cases
from cli.sr_bench.router_flow_gpqa import (
    BENCHMARK,
    EVALSCOPE_REVISION,
    LIMITATIONS,
    NATIVE_BENCHMARK,
    NATIVE_GENERATION,
    NATIVE_SOURCE,
    PROTOCOL,
    SAMPLING,
    SEED,
    SELECTION,
    SMOKE_LIMIT,
)

from .run_evalscope_suite import (
    DEFAULT_SUITE,
    load_suite,
    merge_dicts,
    mode_specific_mapping,
)


def build_manifest(
    *,
    base: dict[str, Any],
    suite_path: Path,
    source_path: Path,
    revision: str,
    limit: int | None = None,
) -> dict[str, Any]:
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Native source revision is required")
    suite = load_suite(suite_path)
    rows = [row for row in suite["benchmarks"] if row["id"] == "gpqa_d"]
    if len(rows) != 1:
        raise ValueError("The suite must contain one gpqa_d row")
    row = rows[0]
    generation = merge_dicts(
        suite["defaults"].get("generation_config", {}),
        mode_specific_mapping(row, "generation_config", "smoke"),
    )
    if (
        row["dataset"] != NATIVE_BENCHMARK
        or row["metric"] != "mean_acc"
        or row["smoke_limit"] != SMOKE_LIMIT
        or row["eval_batch_size"] != NATIVE_GENERATION["batch_size"]
        or generation != NATIVE_GENERATION
        or mode_specific_mapping(row, "dataset_args", "smoke")
        or row.get("judge_strategy")
        or row.get("sandbox")
    ):
        raise ValueError("The suite differs from the supported GPQA smoke protocol")
    if not isinstance(base, dict) or any(
        key in base for key in ("cases", "dataset", "benchmark_options", "plan_sha256")
    ):
        raise ValueError(
            "Use a target/limits manifest without existing benchmark cases"
        )
    manifest = copy.deepcopy(base)
    for key, value in (
        ("version", VERSION),
        ("mode", "live"),
        ("profile", "smoke"),
        ("seed", SEED),
        ("output_policy", "bounded"),
    ):
        if key in manifest and manifest[key] != value:
            raise ValueError(f"{key} conflicts with the GPQA smoke protocol")
        manifest[key] = value
    if {**SAMPLING, **manifest.get("sampling", {})} != SAMPLING:
        raise ValueError("sampling conflicts with the GPQA smoke protocol")
    manifest["sampling"] = dict(SAMPLING)
    limits = manifest.get("limits", {})
    required_limits = {
        "concurrency": NATIVE_GENERATION["batch_size"],
        "total_timeout_s": NATIVE_GENERATION["timeout"],
        "max_calls_per_case": 1,
    }
    if any(
        key in limits and limits[key] != value for key, value in required_limits.items()
    ):
        raise ValueError("limits conflict with the GPQA smoke protocol")
    manifest["limits"] = {
        **DEFAULT_LIMITS,
        "max_output_tokens": SAMPLING["max_tokens"],
        "case_timeout_s": NATIVE_GENERATION["timeout"],
        **limits,
        **required_limits,
    }
    if (
        manifest["limits"]["max_output_tokens"] < SAMPLING["max_tokens"]
        or manifest["limits"]["case_timeout_s"] < NATIVE_GENERATION["timeout"]
        or manifest["limits"]["max_run_seconds"] < NATIVE_GENERATION["timeout"]
    ):
        raise ValueError("limits cannot accommodate the GPQA generation contract")
    selected_limit = SMOKE_LIMIT if limit is None else limit
    cases = prepare_cases(source_path=source_path, limit=selected_limit)
    manifest["cases"] = cases
    manifest["benchmark_options"] = {
        BENCHMARK: {
            "protocol": PROTOCOL,
            "source_revision": EVALSCOPE_REVISION,
            "native_generation": generation,
            "selection": SELECTION,
            "limit": selected_limit,
            "custom_subset": selected_limit < SMOKE_LIMIT,
            "suite_sha256": hashlib.sha256(
                read_small(suite_path, MAX_SCAN_BYTES)
            ).hexdigest(),
            "source": {
                "url": NATIVE_SOURCE,
                "revision": revision,
                "sha256": cases[0]["metadata"]["source_sha256"],
                "revision_verification": "operator-declared-with-content-digest",
            },
        }
    }
    manifest["limitations"] = [*manifest.get("limitations", []), *LIMITATIONS]
    if selected_limit < SMOKE_LIMIT:
        manifest["limitations"].append(
            f"Custom smoke subset: the first {selected_limit} of {SMOKE_LIMIT} native smoke cases."
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an EvalScope-derived GPQA smoke plan without model requests."
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        base=load_document(args.manifest),
        suite_path=args.suite,
        source_path=args.source_path,
        revision=args.revision,
        limit=args.limit,
    )
    with args.output.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
