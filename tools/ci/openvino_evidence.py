#!/usr/bin/env python3
"""Attest published inference and repeated, same-process OpenVINO ownership tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ci_results import actual_platform

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_TESTS = {
    "TestOwnedOpenVINORuntimeIntegration",
    "TestOpenVINOQualifiedFixtureMetadata",
}
REPEATS = 3


def events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]


def discovered_package(path: Path, required: set[str]) -> str:
    listed = events(path)
    names = [
        (event["Package"], event["Output"].strip())
        for event in listed
        if re.fullmatch(r"Test\w+", event.get("Output", "").strip())
    ]
    packages = {package for package, _ in names}
    if (
        not required
        or len(packages) != 1
        or len(names) != len(set(names))
        or {name for _, name in names} != required
    ):
        raise ValueError(f"incomplete or duplicate OpenVINO discovery: {path}")
    package = packages.pop()
    terminals = [
        (event.get("Package"), event["Action"])
        for event in listed
        if not event.get("Test") and event.get("Action") in {"pass", "fail", "skip"}
    ]
    if terminals != [(package, "pass")]:
        raise ValueError(f"OpenVINO discovery did not finish successfully: {path}")
    return package


def run_cases(
    path: Path, package: str, required: set[str], repeats: int, prefix: str
) -> tuple[list[dict], list[str]]:
    """Keep each real -count iteration, requiring complete Go run/terminal pairs."""
    if not required or repeats < 1:
        raise ValueError("OpenVINO execution requires a nonempty repeated inventory")
    counts = dict.fromkeys(required, 0)
    active: dict[str, int] = {}
    started: set[str] = set()
    cases, terminals = [], []
    for event in events(path):
        name, action = event.get("Test", ""), event.get("Action")
        if not name:
            if action in {"pass", "fail", "skip"}:
                if active or any(count != repeats for count in counts.values()):
                    raise ValueError("OpenVINO process ended with incomplete tests")
                terminals.append((event.get("Package"), action))
            continue
        if action not in {"run", "pass", "fail", "skip"}:
            continue
        root = name.split("/", 1)[0]
        if event.get("Package") != package or root not in required:
            raise ValueError(f"unexpected OpenVINO test: {name}")
        if action == "run":
            if name == root:
                if name in active or counts[root] >= repeats:
                    raise ValueError(f"duplicate or excess OpenVINO repeat: {name}")
                counts[root] += 1
            elif root not in active:
                raise ValueError(f"OpenVINO subtest has no active parent: {name}")
            iteration = counts[root]
            identity = f"{prefix}/run-{iteration}/{package}/{name}"
            if identity in started:
                raise ValueError(f"duplicate OpenVINO run: {identity}")
            started.add(identity)
            active[name] = iteration
            continue
        if name not in active:
            raise ValueError(f"OpenVINO terminal has no active run: {name}")
        iteration = active.pop(name)
        if action != "pass":
            raise ValueError(f"required OpenVINO test did not pass: {name} ({action})")
        if any(child.startswith(name + "/") for child in active):
            raise ValueError(
                f"OpenVINO parent finished with incomplete subtests: {name}"
            )
        cases.append(
            {
                "id": f"{prefix}/run-{iteration}/{package}/{name}",
                "status": "passed",
            }
        )
    if active or any(count != repeats for count in counts.values()):
        raise ValueError(f"incomplete OpenVINO repeats: {path}")
    if terminals != [(package, "pass")]:
        raise ValueError(f"OpenVINO process did not finish successfully: {path}")
    expected = [
        f"{prefix}/run-{iteration}/{package}/{name}"
        for iteration in range(1, repeats + 1)
        for name in sorted(required)
    ]
    return cases, expected


def evidence(directory: Path) -> dict:
    report = json.loads((directory / "inference.json").read_text())
    if (
        report.get("passed") is not True
        or report.get("provider") != "openvino"
        or report.get("platform") != actual_platform()
    ):
        raise ValueError("OpenVINO inference evidence does not match its execution")
    package = (ROOT / "openvino-binding/go.mod").read_text().splitlines()[0].split()[1]
    cases, expected = run_cases(
        directory / "tests.jsonl", package, {"TestPublishedOpenVINO"}, 1, "published"
    )
    expected.extend(
        f"published/run-1/{package}/TestPublishedOpenVINO/{name}"
        for name in ("domain", "embedding")
    )
    if set(expected) - {case["id"] for case in cases}:
        raise ValueError("missing required OpenVINO published inference")
    bindings = set(
        re.findall(
            r"^func (TestOwned\w+)\(",
            (ROOT / "openvino-binding/owned_model_test.go").read_text(),
            re.MULTILINE,
        )
    )
    for group, required in (
        ("owned-bindings", bindings),
        ("owned-runtime", RUNTIME_TESTS),
    ):
        path = directory / group
        package = discovered_package(path / "discovery.jsonl", required)
        measured, selected = run_cases(
            path / "tests.jsonl", package, required, REPEATS, group
        )
        cases.extend(measured)
        expected.extend(selected)
    return {
        "runtime": "openvino",
        "device": report["device"].lower(),
        "platform": report["platform"],
        "models": report["models"],
        "cases": cases,
        "expected_cases": expected,
    }
