#!/usr/bin/env python3
"""Reconcile actual execution receipts with the complete pre-execution CI plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ci_plan import digest
from ci_results import collection_errors, execution_errors
from verification_catalog import full_cpu_ids


@dataclass(frozen=True)
class GateVerdict:
    required: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def evaluate_gate(
    plan: dict[str, Any],
    receipts: list[dict],
    *,
    builds: list[dict] | None = None,
    jobs: dict | None = None,
) -> GateVerdict:
    errors: list[str] = []
    required = plan.get("expected_verification_ids", [])
    if not isinstance(required, list) or len(set(required)) != len(required):
        return GateVerdict((), ("plan verification IDs are invalid or duplicated",))
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("schema_version") != 1 or plan.get("plan_sha256") != digest(unsigned):
        errors.append("plan schema or content digest is invalid")
    planned = {record["id"]: record for record in plan.get("verifications", [])}
    if len(planned) != len(plan.get("verifications", [])) or set(planned) != set(
        required
    ):
        errors.append("plan records do not exactly match expected verification IDs")
    if plan.get("full_cpu") and not (plan.get("draft") and plan.get("profile") == "pr"):
        missing = set(full_cpu_ids()) - set(required)
        if missing:
            errors.append(f"full CPU required inventory omitted: {sorted(missing)}")
    if not required and not (plan.get("draft") and plan.get("profile") == "pr"):
        errors.append("non-draft plan has no required verifications")
    actual = {}
    for receipt in receipts:
        name = receipt.get("id")
        if name in actual:
            errors.append(f"duplicate result for {name}")
        actual[name] = receipt
    if set(actual) - set(required):
        errors.append(
            f"unplanned verification results: {sorted(set(actual) - set(required))}"
        )
    build_map = {}
    for build in builds or []:
        name = build.get("id")
        if name in build_map:
            errors.append(f"duplicate build identity {name}")
        build_map[name] = build
        if build.get("source_sha") != plan.get("source_sha"):
            errors.append(f"build {name}: source SHA differs")
    expected_builds = {f"image:{name}" for name in plan.get("images", [])}
    if plan.get("native"):
        expected_builds.add("native:cpu")
    if set(build_map) != expected_builds:
        errors.append(
            f"build inventory mismatch: missing={sorted(expected_builds-set(build_map))}, extra={sorted(set(build_map)-expected_builds)}"
        )
    for name in required:
        receipt, record = actual.get(name), planned.get(name)
        if not receipt or not record:
            errors.append(f"required verification {name}: missing")
            continue
        if receipt.get("result") != "success":
            errors.append(
                f"required verification {name}: {receipt.get('result', 'missing status')}"
            )
        for key in (
            "source_sha",
            "contract_sha256",
            "inventory",
            "runtime",
            "device",
            "platform",
        ):
            if receipt.get(key) != record.get(key):
                errors.append(f"{name}: {key} differs from plan")
        evidence = receipt.get("evidence", {})
        errors.extend(
            f"{name}: {error}"
            for error in collection_errors(evidence, record["activity"])
        )
        if receipt.get("evidence_sha256") != digest(evidence):
            errors.append(f"{name}: evidence digest differs")
        for key in ("runtime", "device", "platform"):
            if evidence.get(key) != record[key]:
                errors.append(f"{name}: evidence {key} differs from plan")
        if receipt.get("execution") != record.get("execution"):
            errors.append(f"{name}: execution mode differs from plan")
        producer = receipt.get("execution", {}).get(
            "host_platform", receipt.get("platform")
        )
        errors.extend(
            f"{name}: {error}" for error in execution_errors(record, evidence, producer)
        )
        consumed = receipt.get("artifacts", [])
        consumed_ids = [item.get("id") for item in consumed]
        if len(set(consumed_ids)) != len(consumed_ids):
            errors.append(f"{name}: duplicate consumed artifact identity")
        dependencies = {f"image:{image}" for image in record["images"]}
        if record["native"]:
            dependencies.add("native:cpu")
        if not dependencies <= set(consumed_ids):
            errors.append(f"{name}: missing artifact dependencies")
        for artifact in consumed:
            build = build_map.get(artifact.get("id"))
            if not build or artifact.get("sha256") != build.get("sha256"):
                errors.append(
                    f"{name}: consumed artifact {artifact.get('id')} differs from build"
                )
    if jobs is not None:
        expected_jobs = {"plan"} | {record["executor"] for record in planned.values()}
        if plan.get("images"):
            expected_jobs.add("images")
        if plan.get("native"):
            expected_jobs.add("native-build")
        for name, record in jobs.items():
            if record.get("result") in {"failure", "cancelled"}:
                errors.append(f"prerequisite {name}: {record['result']}")
        for name in expected_jobs:
            if jobs.get(name, {}).get("result") != "success":
                errors.append(
                    f"required executor {name}: {jobs.get(name, {}).get('result', 'missing')}"
                )
    return GateVerdict(tuple(required), tuple(sorted(set(errors))))


def load_builds(directory: Path) -> list[dict]:
    builds = []
    for path in sorted(directory.glob("ci-build-image-*/manifest.json")):
        record = json.loads(path.read_text())
        if not any(
            image.get("platform") == "linux/amd64" for image in record.get("images", [])
        ):
            raise ValueError(f"{path}: missing actual Linux AMD64 image")
        builds.append(
            {
                "id": f"image:{record['id']}",
                "source_sha": record["source_sha"],
                "sha256": record["sha256"],
            }
        )
    native = directory / "ci-build-native-cpu/receipt.json"
    if native.exists():
        manifest = native.with_name("manifest.json")
        rows = json.loads(native.read_text())
        if (
            len(rows) != 1
            or rows[0].get("sha256")
            != hashlib.sha256(manifest.read_bytes()).hexdigest()
        ):
            raise ValueError("native build receipt differs from actual manifest")
        if json.loads(manifest.read_text()).get("platform") != "linux/amd64":
            raise ValueError("native build platform is not Linux AMD64")
        builds.extend(rows)
    return builds


def render_summary(verdict: GateVerdict) -> str:
    lines = [
        f"CI gate {'passed' if verdict.passed else 'failed'}; {len(verdict.required)} required verifications."
    ]
    lines.extend(f"- {error}" for error in verdict.errors)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        if os.environ.get("GITHUB_SHA", plan.get("source_sha")) != plan.get(
            "source_sha"
        ):
            raise ValueError("plan source SHA differs from this workflow run")
        receipts = [
            json.loads(path.read_text())
            for path in sorted(args.results.glob("ci-result-*/*.json"))
        ]
        jobs = (
            json.loads(os.environ["EXECUTOR_RESULTS"])
            if "EXECUTOR_RESULTS" in os.environ
            else None
        )
        verdict = evaluate_gate(
            plan, receipts, builds=load_builds(args.results), jobs=jobs
        )
    except (ValueError, TypeError, KeyError, OSError) as exc:
        verdict = GateVerdict((), (str(exc),))
    summary = render_summary(verdict)
    print(summary, end="")
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a") as stream:
            stream.write(summary)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
