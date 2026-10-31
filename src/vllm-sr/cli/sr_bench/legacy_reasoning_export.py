from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .dataset_io import read_small
from .external import _preflight_harness
from .legacy_reasoning import (
    ALIASES,
    CONFIGURATIONS,
    ENVIRONMENT,
    MAX_EXPORT_BYTES,
    SOURCE_REVISION,
    configuration,
    worker_request,
)
from .setup import harness_paths


def export_manifest(
    *,
    manifest: dict[str, Any],
    dataset: str,
    source_bundle: str | Path,
    seed: int = 42,
    categories: list[str] | None = None,
    samples_per_category: int | None = None,
) -> dict[str, Any]:
    if "cases" in manifest or "dataset" in manifest:
        raise ValueError("The run template must not already contain cases or a dataset")
    if manifest.get("profile", "quick") not in {"smoke", "quick"}:
        raise ValueError("Legacy plain diagnostics do not establish a holdout")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Native selection seed must be an unsigned 32-bit integer")
    if samples_per_category is not None and (
        type(samples_per_category) is not int or samples_per_category <= 0
    ):
        raise ValueError("samples_per_category must be a positive integer")
    if categories is not None and (
        not categories
        or not all(isinstance(value, str) and value for value in categories)
    ):
        raise ValueError("categories must contain nonempty native category names")
    name = configuration(dataset)
    _preflight_harness(
        benchmark=ENVIRONMENT, config={"source_revision": SOURCE_REVISION}
    )
    root, interpreter = harness_paths(ENVIRONMENT)
    exported = worker_request(
        operation="export",
        payload={
            "dataset": name,
            "source_bundle": str(Path(source_bundle).expanduser().resolve()),
            "seed": seed,
            "categories": categories,
            "samples_per_category": samples_per_category,
        },
        root=root,
        interpreter=interpreter,
    )
    options: dict[str, Any] = {
        **manifest.get("benchmark_options", {}),
        **exported["benchmark_options"],
    }
    return {
        **manifest,
        "seed": seed,
        "profile": manifest.get("profile", "quick"),
        "cases": exported["cases"],
        "benchmark_options": options,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export local native plain reasoning cases into an sr-bench run."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(CONFIGURATIONS.keys() | ALIASES.keys()),
        required=True,
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON run template with targets and limits.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--samples-per-category", type=int)
    args = parser.parse_args()
    exported = export_manifest(
        manifest=json.loads(read_small(args.manifest, MAX_EXPORT_BYTES)),
        dataset=args.dataset,
        source_bundle=args.source_bundle,
        seed=args.seed,
        categories=args.categories,
        samples_per_category=args.samples_per_category,
    )
    content = (json.dumps(exported, indent=2, ensure_ascii=False) + "\n").encode()
    if len(content) > MAX_EXPORT_BYTES:
        raise ValueError("Legacy reasoning export exceeds the supported byte limit")
    with os.fdopen(
        os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as output:
        output.write(content)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "cases": len(exported["cases"]),
                "model_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
