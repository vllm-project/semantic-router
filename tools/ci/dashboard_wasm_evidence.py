"""Require actual Go/js-WASM tests and every source-declared Node assertion."""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections import Counter
from pathlib import Path


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def go_evidence(inventory: Path, execution: Path) -> tuple[list[dict], list[str]]:
    listed = events(inventory)
    expected = [
        f"go-js-wasm:{row['Package']}/{row['Output'].strip()}"
        for row in listed
        if re.fullmatch(r"(?:Test|Fuzz|Example)\w*", row.get("Output", "").strip())
    ]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("empty or duplicate Go WASM discovery")
    expected_packages = {row["Package"] for row in listed if row.get("Package")}
    expected_roots = set(expected)
    started: Counter = Counter()
    passed: Counter = Counter()
    packages: Counter = Counter()
    roots: set[str] = set()
    for row in events(execution):
        action, test = row.get("Action"), row.get("Test")
        if action in {"fail", "skip"}:
            raise ValueError(f"Go WASM did not pass: {row.get('Package')}/{test}")
        if not test:
            if action == "pass":
                packages[row["Package"]] += 1
            continue
        identity = f"go-js-wasm:{row['Package']}/{test}"
        root = f"go-js-wasm:{row['Package']}/{test.split('/', 1)[0]}"
        if root not in expected_roots:
            raise ValueError(f"undiscovered Go WASM case: {identity}")
        if action == "run":
            started[identity] += 1
            if started[identity] != 1:
                raise ValueError(f"duplicate Go WASM execution: {identity}")
        elif action == "pass":
            passed[identity] += 1
            if "/" not in test:
                roots.add(identity)
            if started[identity] != 1 or passed[identity] != 1:
                raise ValueError(f"invalid Go WASM completion: {identity}")
    if packages != Counter(dict.fromkeys(expected_packages, 1)):
        raise ValueError("Go WASM package did not complete successfully")
    if started != passed:
        raise ValueError("incomplete Go WASM test lifecycle")
    if roots != expected_roots:
        raise ValueError("Go WASM execution differs from discovery")
    cases = [{"id": name, "status": "passed"} for name in sorted(passed)]
    # Subtests are checked individually; the independently discovered top-level
    # cases remain the required inventory, as in the other Go adapters.
    return cases, expected


def node_evidence(source: Path, execution: Path) -> tuple[list[dict], list[str]]:
    content = source.read_text()
    calls = re.findall(r"(?m)^\s*assert\(", content)
    names = re.findall(r'(?m)^\s*assert\(\s*"([a-z][a-z0-9_]*)"\s*,', content)
    if not names or len(names) != len(calls) or len(names) != len(set(names)):
        raise ValueError("Node WASM assertions need unique literal source IDs")
    expected = ["node-wasm:" + name for name in names]
    cases = []
    for row in events(execution):
        if set(row) != {"id", "status"} or row["status"] != "passed":
            raise ValueError("Node WASM assertion did not pass")
        cases.append({"id": "node-wasm:" + row["id"], "status": row["status"]})
    if Counter(case["id"] for case in cases) != Counter(expected):
        raise ValueError("Node WASM execution differs from source inventory")
    return cases, expected


def evidence(report: Path, node_source: Path) -> dict:
    go_cases, go_expected = go_evidence(
        report / "inventory.jsonl", report / "go-tests.jsonl"
    )
    node_cases, node_expected = node_evidence(node_source, report / "node-tests.jsonl")
    machine = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine(), platform.machine()
    )
    return {
        "runtime": "none",
        "device": "none",
        "platform": f"{platform.system().lower()}/{machine}",
        "artifacts": [],
        "cases": go_cases + node_cases,
        "expected_cases": go_expected + node_expected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--node-source", type=Path, required=True)
    args = parser.parse_args()
    result = evidence(args.report, args.node_source)
    (args.report / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Dashboard WASM: {len(result['cases'])} actual tests/assertions passed")
