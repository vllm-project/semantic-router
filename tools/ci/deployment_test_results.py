#!/usr/bin/env python3
"""Normalize actual Kubernetes and operator test evidence, rejecting omissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ci_results import actual_platform, artifact_records


def require_complete(cases: list[dict], expected: list[str]) -> None:
    ids = [case["id"] for case in cases]
    if (
        not expected
        or any(not name for name in expected)
        or len(expected) != len(set(expected))
    ):
        raise ValueError("expected deployment test inventory is empty or duplicated")
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(
            "deployment test executions differ from the expected inventory"
        )
    if any(case["status"] != "passed" for case in cases):
        raise ValueError("required deployment test failed or skipped")


def kubernetes(report: dict, profile: str) -> dict:
    if report["profile"] != profile:
        raise ValueError("Kubernetes report belongs to a different profile")
    expected = report["expected_cases"]
    cases = [
        {"id": row["Name"], "status": "passed" if row["Passed"] is True else "failed"}
        for row in report["test_results"]
    ]
    require_complete(cases, expected)
    if report["status"] != "PASSED" or report["exit_code"] != 0:
        raise ValueError("Kubernetes framework reported a failed run")
    if (
        report["total_tests"] != len(cases)
        or report["passed_tests"] != len(cases)
        or report["failed_tests"] != 0
    ):
        raise ValueError("Kubernetes report counters disagree with actual test results")
    return {"cases": cases, "expected_cases": expected}


def go_unit(discovery: list[dict], events: list[dict]) -> dict:
    expected = []
    for row in discovery:
        if row.get("Action") == "output":
            name = row.get("Output", "").strip()
            if name.startswith("Test") and name.replace("_", "").isalnum():
                expected.append(f"{row['Package']}/{name}")
    cases, roots = [], []
    for row in events:
        name, action = row.get("Test"), row.get("Action")
        if name and action in {"pass", "fail", "skip"}:
            case = {
                "id": f"{row['Package']}/{name}",
                "status": {"pass": "passed", "fail": "failed", "skip": "skipped"}[
                    action
                ],
            }
            cases.append(case)
            if "/" not in name:
                roots.append(case)
    require_complete(roots, expected)
    # Subtests are discovered by Go when their parent runs; retain and check them too.
    executed = [
        f"{row['Package']}/{row['Test']}"
        for row in events
        if row.get("Test") and row.get("Action") == "run"
    ]
    require_complete(cases, executed)
    if any(row.get("Action") == "fail" and not row.get("Test") for row in events):
        raise ValueError("operator Go package failed")
    return {"cases": cases, "expected_cases": executed}


def operator(directory: Path, variants: list[dict], jobs: dict) -> dict:
    for job in ("checks", "bundle-validate", "integration-test"):
        if jobs.get(job, {}).get("result") != "success":
            raise ValueError(f"operator prerequisite did not succeed: {job}")
    names = [row["cache-backend"] for row in variants]
    if not names or len(names) != len(set(names)):
        raise ValueError("operator variant inventory is empty or duplicated")
    unit = json.loads((directory / "operator-unit" / "operator-unit.json").read_text())
    require_complete(unit["cases"], unit["expected_cases"])
    cases = [{**row, "id": "unit/" + row["id"]} for row in unit["cases"]]
    expected = ["unit/" + name for name in unit["expected_cases"]]
    artifacts = json.loads(
        (directory / "operator-bundle" / "artifacts.json").read_text()
    )
    for name in names:
        base = directory / f"operator-request-{name}"
        report = json.loads((base / "operator-request.json").read_text())
        require_complete(report["cases"], ["operator-routed-request"])
        if report["expected_cases"] != ["operator-routed-request"]:
            raise ValueError("unexpected operator request contract")
        cases.extend({**row, "id": name + "/" + row["id"]} for row in report["cases"])
        expected.append(name + "/operator-routed-request")
        artifacts.extend(json.loads((base / "artifacts.json").read_text()))
    require_complete(cases, expected)
    return {
        "cases": cases,
        "expected_cases": expected,
        "artifacts": artifact_records({"artifacts": artifacts}, environ={}),
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("kubernetes", "operator-unit", "operator"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--variants")
    parser.add_argument("--jobs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.kind == "kubernetes":
            evidence = kubernetes(json.loads(args.report.read_text()), args.profile)
        elif args.kind == "operator-unit":
            evidence = go_unit(read_jsonl(args.discovery), read_jsonl(args.events))
        else:
            evidence = operator(
                args.directory, json.loads(args.variants), json.loads(args.jobs)
            )
        evidence.update(
            runtime="none" if args.kind == "operator-unit" else "candle",
            device="none" if args.kind == "operator-unit" else "cpu",
            platform=actual_platform(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Deployment evidence rejected: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
