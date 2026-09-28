#!/usr/bin/env python3
"""Normalize completed workflow activities and framework-owned test reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from ci_results import actual_platform, make_receipt


def junit_cases(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    cases = []
    for item in root.iter("testcase"):
        identity = f"{item.get('classname', '')}::{item.attrib['name']}"
        status = "passed"
        if item.find("skipped") is not None:
            status = "skipped"
        elif item.find("failure") is not None or item.find("error") is not None:
            status = "failed"
        cases.append({"id": identity, "status": status})
    if not cases:
        raise ValueError(f"empty framework report: {path}")
    return cases


def go_cases(
    events_path: Path, inventory_path: Path, exclusions: set[str]
) -> tuple[list[dict], list[str]]:
    events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if line.startswith("{")
    ]
    listed = [
        json.loads(line)
        for line in inventory_path.read_text().splitlines()
        if line.startswith("{")
    ]
    expected = [
        f"{event['Package']}/{event['Output'].strip()}"
        for event in listed
        if re.fullmatch(r"(?:Test|Fuzz|Example)\w*", event.get("Output", "").strip())
        and event["Output"].strip() not in exclusions
    ]
    cases = []
    for event in events:
        name, action = event.get("Test", ""), event.get("Action")
        if name and action in {"pass", "fail", "skip"}:
            if "/" not in name:
                cases.append(
                    {
                        "id": f"{event['Package']}/{name}",
                        "status": {
                            "pass": "passed",
                            "fail": "failed",
                            "skip": "skipped",
                        }[action],
                    }
                )
            elif action != "pass":
                raise ValueError(f"required Go subtest did not pass: {name} ({action})")
    return cases, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification")
    parser.add_argument("--verifications")
    parser.add_argument("--id")
    parser.add_argument("--check", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--junit", type=Path, action="append", default=[])
    parser.add_argument("--go-json", type=Path)
    parser.add_argument("--go-inventory", type=Path)
    parser.add_argument("--exclude-go", action="append", default=[])
    parser.add_argument("--benchmarks", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verification = (
        json.loads(args.verification)
        if args.verification
        else next(
            record
            for record in json.loads(args.verifications)
            if record["id"] == args.id
        )
    )
    evidence = {
        "runtime": "none",
        "device": "none",
        "platform": actual_platform(),
        "artifacts": [],
    }
    if args.check:
        evidence.update(
            checks=[{"id": name, "status": "passed"} for name in args.check],
            expected_checks=args.check,
        )
    else:
        cases, expected = [], []
        for path in args.evidence:
            source = json.loads(path.read_text())
            prefix = path.parent.name + ":"
            cases.extend(
                {**case, "id": prefix + case["id"]} for case in source["cases"]
            )
            expected.extend(prefix + name for name in source["expected_cases"])
        for path in args.junit:
            actual = junit_cases(path)
            cases.extend(
                {**case, "id": path.stem + ":" + case["id"]} for case in actual
            )
            expected.extend(path.stem + ":" + case["id"] for case in actual)
        if args.go_json:
            actual, selected = go_cases(
                args.go_json, args.go_inventory, set(args.exclude_go)
            )
            cases.extend(actual)
            expected.extend(selected)
        if args.benchmarks:
            result = json.loads(args.benchmarks.read_text())
            expected.extend(json.loads(args.inventory.read_text())["benchmarks"])
            for name, metric in result["benchmarks"].items():
                identity = metric.get("model_identity")
                if identity and (identity["provider"], identity["device"]) != (
                    "candle",
                    "cpu",
                ):
                    raise ValueError(
                        "performance evidence requires actual Candle CPU measurements"
                    )
                cases.append({"id": name, "status": "passed", "measurement": metric})
            evidence.update(runtime="candle", device="cpu")
        cases.extend({"id": name, "status": "passed"} for name in args.case)
        expected.extend(args.case)
        evidence.update(cases=cases, expected_cases=expected)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = make_receipt(
        verification,
        evidence,
        source_sha=source_sha,
        execution_platform=actual_platform(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
