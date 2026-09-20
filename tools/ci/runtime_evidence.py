#!/usr/bin/env python3
"""Adapt native, local-stack and Preview framework results for the CI gate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from http import HTTPStatus
from pathlib import Path

from image_calibration import evidence as image_calibration_evidence
from openvino_evidence import evidence as openvino_evidence
from riscv_evidence import evidence as riscv_evidence

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def host_platform() -> str:
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(
        platform.machine(), platform.machine()
    )
    return platform.system().lower() + "/" + arch


def native_evidence(directory: Path) -> dict:
    if (directory / "inference.json").exists():
        return openvino_evidence(directory)
    report = read(directory / "results.json")
    paths = [directory / "results.json"]
    if report["provider"] == "candle":
        paths.append(directory / "multimodal/results.json")
    actual_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    cases, expected, models = [], [], []
    for path in paths:
        suite_report = read(path)
        if (
            suite_report["source_sha"] != actual_sha
            or suite_report["provider"] != report["provider"]
        ):
            raise ValueError("Native evidence source or runtime mismatch")
        models.extend(suite_report["models"])
        for suite in suite_report["suites"]:
            prefix = suite_report["suite"] + ":" + suite["package"] + ":"
            expected.extend(prefix + name for name in suite["expected"])
            for field, status in (
                ("passed", "passed"),
                ("skipped", "skipped"),
                ("failed", "failed"),
            ):
                cases.extend(
                    {"id": prefix + name, "status": status} for name in suite[field]
                )
            # A crashed Go process can lack a terminal test event.
            if suite["exit_code"] and not suite["failed"]:
                cases.append({"id": prefix + "process", "status": "failed"})
    return {
        "runtime": report["provider"],
        "device": report["device"],
        "platform": host_platform(),
        "models": models,
        "cases": cases,
        "expected_cases": expected,
    }


def local_evidence(directory: Path, suite: str) -> dict:
    report = read(directory / f"{suite}-test-report.json")
    return {**report, "runtime": "candle", "device": "cpu", "platform": host_platform()}


def recipe_cases(evaluation: dict, prefix: str) -> list[dict]:
    # Probes are observations in an authored evaluation, with per-decision and
    # stress acceptance policies. Preserve those policies rather than silently
    # replacing them with a new 100% accuracy threshold in the receipt adapter.
    cases = [
        {
            "id": prefix + "request:" + item["id"],
            "status": (
                "passed"
                if HTTPStatus.OK
                <= item.get("http_status", 0)
                < HTTPStatus.MULTIPLE_CHOICES
                and not item.get("error")
                else "failed"
            ),
            "matched": item["matched"],
        }
        for item in evaluation["results"]
    ]
    cases.append(
        {
            "id": prefix + "acceptance",
            "status": (
                "passed"
                if evaluation["passed"] and evaluation["execution"]["complete"]
                else "failed"
            ),
        }
    )
    return cases


def recipe_evidence(directory: Path) -> dict:
    sys.path.insert(0, str(ROOT / "tools/calibration/recipe"))
    from recipe_conformance import (  # noqa: PLC0415 - only Preview needs calibration tooling
        cpu_inventory,
        discover_inventory,
        load_probe_manifest,
    )
    from recipe_conformance_sources import (  # noqa: PLC0415
        discover_recipe_sources,
        shard_inventory,
    )

    cases, expected, identities = [], [], []
    for source in discover_recipe_sources(ROOT / "config/recipes"):
        inventory = cpu_inventory(discover_inventory(source.recipes_root))
        for index, _ in enumerate(shard_inventory(inventory, 3)):
            identities.extend(read(directory / f"image-{source.name}-{index}.json"))
        for recipe in inventory:
            _, probes = load_probe_manifest(
                source.recipes_root / recipe.name / "probes.yaml"
            )
            prefix = source.name + ":" + recipe.name + ":"
            expected.extend(prefix + "request:" + probe.probe_id for probe in probes)
            expected.append(prefix + "acceptance")
            report = read(
                directory / source.report_subdir / recipe.name / "eval-report.json"
            )
            cases.extend(recipe_cases(report["evaluation"], prefix))
    return {
        "runtime": "candle",
        "device": "cpu",
        "platform": host_platform(),
        "cases": cases,
        "expected_cases": expected,
        "artifacts": identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=["native", "local", "recipes", "image-calibration", "riscv-qemu"],
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=["cli", "memory"], default="cli")
    args = parser.parse_args()
    if args.kind == "image-calibration":
        result = image_calibration_evidence(args.directory)
    elif args.kind == "riscv-qemu":
        result = riscv_evidence(args.directory)
    elif args.kind == "native":
        result = native_evidence(args.directory)
    elif args.kind == "local":
        result = local_evidence(args.directory, args.suite)
    else:
        result = recipe_evidence(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
