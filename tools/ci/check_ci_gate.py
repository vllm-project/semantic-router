#!/usr/bin/env python3
"""Reconcile actual execution receipts with the complete pre-execution CI plan."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ci_plan import component_batches, digest
from ci_results import collection_errors, execution_errors
from classify_pr_changes import classify
from domain_registry import load_domain_registry
from execution_batches import (
    dispatch_job,
    e2e_batches,
    expected_dispatch_jobs,
    image_producers,
    platform_batches,
)
from execution_targets import TARGET_FIELDS, target_errors
from provider_mocker_image import validate_acquisition
from release_guard_waiver import planned_waiver
from verification_catalog import full_cpu_ids, load_catalog, verification_records


@dataclass(frozen=True)
class GateVerdict:
    required: tuple[str, ...]
    errors: tuple[str, ...]
    deferred_verifications: tuple[str, ...] = ()
    deferred_images: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.errors:
            return "failure"
        if self.deferred_verifications or self.deferred_images:
            return "awaiting-approval"
        return "success"

    @property
    def passed(self) -> bool:
        return self.status == "success"


def valid_inventory(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(name, str) and name for name in value)
        and len(set(value)) == len(value)
    )


def evaluate_gate(
    plan: dict[str, Any],
    receipts: list[dict],
    *,
    builds: list[dict] | None = None,
    jobs: dict | None = None,
    require_actions: bool = False,
) -> GateVerdict:
    errors: list[str] = []
    required = plan.get("expected_verification_ids", [])
    if not valid_inventory(required):
        return GateVerdict((), ("plan verification IDs are invalid or duplicated",))
    deferred = plan.get("deferred_verifications", [])
    complete = plan.get("required_verification_ids", required)
    if (
        not valid_inventory(deferred)
        or not valid_inventory(complete)
        or set(required) & set(deferred)
        or set(complete) != set(required) | set(deferred)
    ):
        return GateVerdict(
            tuple(complete) if valid_inventory(complete) else tuple(required),
            ("required execution and deferred inventories do not partition the plan",),
        )
    images = plan.get("images", [])
    deferred_images = plan.get("deferred_images", [])
    if (
        not valid_inventory(images)
        or not valid_inventory(deferred_images)
        or set(images) & set(deferred_images)
    ):
        return GateVerdict(
            tuple(complete),
            ("execution and deferred image inventories are invalid or overlap",),
            tuple(deferred),
        )
    if (deferred or deferred_images) and not plan.get("approval_required"):
        errors.append("deferred qualification requires an approval-gated plan")
    if deferred:
        records = verification_records(load_domain_registry())
        ineligible = [
            name
            for name in deferred
            if records.get(name, {}).get("execution") != "approved"
        ]
        if ineligible:
            errors.append(
                f"only approval-gated verifications may be deferred: {ineligible}"
            )
    if plan.get("approval_required"):
        selection = classify(plan.get("paths", []), full=bool(plan.get("full_cpu")))
        selected = [] if plan.get("draft") else list(selection.selected_jobs)
        if set(complete) != set(selected):
            errors.append(
                "required qualification inventory differs from changed-path selection"
            )
        selected_images = set() if plan.get("draft") else set(selection.pr_images)
        if set(deferred_images) != selected_images - set(images):
            errors.append(
                "deferred image inventory differs from changed-path selection"
            )
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("schema_version") != 1 or plan.get("plan_sha256") != digest(unsigned):
        errors.append("plan schema or content digest is invalid")
    planned = {record["id"]: record for record in plan.get("verifications", [])}
    if len(planned) != len(plan.get("verifications", [])) or set(planned) != set(
        required
    ):
        errors.append("plan records do not exactly match expected verification IDs")
    for name, record in planned.items():
        errors.extend(f"{name}: {error}" for error in target_errors(record))
        if record.get("contract_sha256") != digest(
            {key: value for key, value in record.items() if key != "contract_sha256"}
        ):
            errors.append(f"{name}: verification contract digest differs")
        if record.get("known_issue_waiver") != planned_waiver(
            plan.get("profile", ""), name
        ):
            errors.append(f"{name}: known-issue waiver differs from release policy")
    if plan.get("full_cpu") and not (plan.get("draft") and plan.get("profile") == "pr"):
        missing = set(full_cpu_ids()) - set(complete)
        if missing:
            errors.append(f"full CPU required inventory omitted: {sorted(missing)}")
    if not complete and not (plan.get("draft") and plan.get("profile") == "pr"):
        errors.append("non-draft plan has no required verifications")
    try:
        if plan.get("expected_dispatch_jobs") != expected_dispatch_jobs(plan):
            errors.append(
                "plan dispatch inventory differs from required contracts and artifacts"
            )
        if plan.get("image_producers") != image_producers(plan["images"]):
            errors.append("plan image producer inventory differs")
        for record in planned.values():
            if record.get("dispatch_job") != dispatch_job(record):
                errors.append(f"{record['id']}: dispatch identity differs")
        for field, build_batches in (
            ("component_batches", component_batches),
            ("platform_batches", platform_batches),
            ("e2e_batches", e2e_batches),
        ):
            if plan.get(field) != build_batches(plan["verifications"]):
                errors.append(
                    f"plan {field} differs from required verification inventory"
                )
    except (ValueError, KeyError, TypeError) as error:
        errors.append(f"invalid dispatch plan: {error}")
    if plan.get("full_cpu_version") != load_catalog()["full_cpu"]["version"]:
        errors.append("plan full CPU inventory version differs")
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
        if name == "image:provider-mocker":
            expected = plan.get("image_sources", {}).get("provider-mocker", {})
            if build.get("inputs_sha256") != expected.get("inputs_sha256"):
                errors.append("provider-mocker build inputs differ from plan")
            if build.get("acquisition") != expected.get("source"):
                errors.append("provider-mocker acquisition mode differs from plan")
            if expected.get("source") == "published":
                try:
                    validate_acquisition(build, expected)
                except ValueError as error:
                    errors.append(str(error))
    expected_builds = {f"image:{name}" for name in plan.get("images", [])}
    if set(build_map) != expected_builds:
        errors.append(
            f"build inventory mismatch: missing={sorted(expected_builds - set(build_map))}, extra={sorted(set(build_map) - expected_builds)}"
        )
    for name in required:
        receipt, record = actual.get(name), planned.get(name)
        if not receipt or not record:
            errors.append(f"required verification {name}: missing")
            continue
        waived = receipt.get("result") == "qualified-with-waiver"
        waiver = record.get("known_issue_waiver") if waived else None
        if waived and (
            plan.get("profile") != "release"
            or waiver != planned_waiver("release", name)
            or not waiver
        ):
            errors.append(f"{name}: qualified-with-waiver is not allowed")
            waiver = None
        elif receipt.get("result") != "success" and not waived:
            errors.append(
                f"required verification {name}: {receipt.get('result', 'missing status')}"
            )
        for key in (
            "source_sha",
            "contract_sha256",
            "inventory",
            "runtime",
            "device",
            *TARGET_FIELDS,
        ):
            if receipt.get(key) != record.get(key):
                errors.append(f"{name}: {key} differs from plan")
        evidence = receipt.get("evidence", {})
        errors.extend(
            f"{name}: {error}"
            for error in collection_errors(evidence, record["activity"], waiver=waiver)
        )
        if receipt.get("evidence_sha256") != digest(evidence):
            errors.append(f"{name}: evidence digest differs")
        for key in ("runtime", "device", "platform"):
            if evidence.get(key) != record[key]:
                errors.append(f"{name}: evidence {key} differs from plan")
        errors.extend(
            f"{name}: {error}"
            for error in execution_errors(
                record,
                receipt.get("platform"),
                receipt.get("execution"),
                require_actions=require_actions,
            )
        )
        consumed = receipt.get("artifacts", [])
        consumed_ids = [item.get("id") for item in consumed]
        if len(set(consumed_ids)) != len(consumed_ids):
            errors.append(f"{name}: duplicate consumed artifact identity")
        dependencies = {f"image:{image}" for image in record["images"]}
        if not dependencies <= set(consumed_ids):
            errors.append(f"{name}: missing artifact dependencies")
        for artifact in consumed:
            build = build_map.get(artifact.get("id"))
            if not build or artifact.get("sha256") != build.get("sha256"):
                errors.append(
                    f"{name}: consumed artifact {artifact.get('id')} differs from build"
                )
    if jobs is not None:
        expected_jobs = plan.get("expected_dispatch_jobs", [])
        for name, record in jobs.items():
            if record.get("result") in {"failure", "cancelled"}:
                errors.append(f"prerequisite {name}: {record['result']}")
        for name in expected_jobs:
            if jobs.get(name, {}).get("result") != "success":
                errors.append(
                    f"required executor {name}: {jobs.get(name, {}).get('result', 'missing')}"
                )
    return GateVerdict(
        tuple(complete),
        tuple(sorted(set(errors))),
        tuple(deferred),
        tuple(deferred_images),
    )


def load_receipts(directory: Path, plan: dict) -> list[dict]:
    # A rerun can expand authorization and regroup workers. Artifacts from an
    # obsolete batch remain in the run even with upload-artifact overwrite.
    # Select physical containers from this plan; never deduplicate their cases.
    names = {
        f"ci-result-{record['id']}"
        for record in plan["verifications"]
        if record["executor"] not in {"tools", "platform", "e2e"}
    }
    names.update(
        f"ci-result-components-{batch['id']}" for batch in plan["component_batches"]
    )
    names.update(
        f"ci-result-{batch['id']}"
        for batch in [*plan["platform_batches"], *plan["e2e_batches"]]
    )
    return [
        json.loads(path.read_text())
        for name in sorted(names)
        for path in sorted((directory / name).glob("*.json"))
    ]


def load_builds(directory: Path, images: list[str] | None = None) -> list[dict]:
    builds = []
    selected = None if images is None else {f"ci-build-image-{name}" for name in images}
    for path in sorted(directory.glob("ci-build-image-*/manifest.json")):
        if selected is not None and path.parent.name not in selected:
            continue
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
                **{
                    key: record[key]
                    for key in (
                        "acquisition",
                        "inputs_sha256",
                        "registry_digest",
                        "image_source_sha",
                        "ref",
                        "images",
                    )
                    if key in record
                },
            }
        )
    return builds


def render_summary(verdict: GateVerdict) -> str:
    headings = {
        "success": "CI gate passed",
        "awaiting-approval": "Waiting for committer approval",
        "failure": "CI gate failed",
    }
    lines = [
        f"## {headings[verdict.status]}",
        "",
        f"{len(verdict.required)} required verifications in the complete plan; "
        f"{len(verdict.required) - len(verdict.deferred_verifications)} scheduled; "
        f"{len(verdict.deferred_verifications)} awaiting approval.",
    ]
    if verdict.errors:
        lines.extend(["", "Validation errors:", ""])
        lines.extend(f"- {error}" for error in verdict.errors)
    if verdict.deferred_verifications or verdict.deferred_images:
        if verdict.status == "awaiting-approval":
            lines.extend(
                [
                    "",
                    "All scheduled verifications passed; qualification remains incomplete.",
                ]
            )
        lines.extend(
            [
                "",
                "A committer with repository write, maintain, or admin permission can "
                "comment `/ci run` on the pull request to run the remaining affected checks.",
            ]
        )
        if verdict.deferred_verifications:
            lines.extend(["", "Verifications awaiting approval:", ""])
            lines.extend(f"- `{name}`" for name in verdict.deferred_verifications)
        if verdict.deferred_images:
            lines.extend(["", "Images awaiting approval:", ""])
            lines.extend(f"- `{name}`" for name in verdict.deferred_images)
    return "\n".join(lines) + "\n"


def gate_report(plan: dict, verdict: GateVerdict) -> dict:
    # Keep the transport proportional to the verification inventory, never the
    # changed-path list, complete contracts, logs, or case evidence.
    return {
        "schema_version": 1,
        "source_sha": plan.get("source_sha", ""),
        "base_sha": plan.get("base_sha", ""),
        "plan_sha256": plan.get("plan_sha256", ""),
        "run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0")),
        "status": verdict.status,
        "required_verification_ids": list(verdict.required),
        "deferred_verifications": list(verdict.deferred_verifications),
        "deferred_images": list(verdict.deferred_images),
        "errors": list(verdict.errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--allow-deferred",
        action="store_true",
        help="Exit successfully while awaiting approval, but never on validation errors.",
    )
    parser.add_argument("--output", type=Path, help="Write the compact CI gate report.")
    args = parser.parse_args()
    plan = {}
    try:
        plan = json.loads(args.plan.read_text())
        if not isinstance(plan, dict):
            raise ValueError("CI plan must be a JSON object")
        if os.environ.get("GITHUB_SHA", plan.get("source_sha")) != plan.get(
            "source_sha"
        ):
            raise ValueError("plan source SHA differs from this workflow run")
        receipts = load_receipts(args.results, plan)
        jobs = (
            json.loads(os.environ["EXECUTOR_RESULTS"])
            if "EXECUTOR_RESULTS" in os.environ
            else None
        )
        verdict = evaluate_gate(
            plan,
            receipts,
            builds=load_builds(args.results, plan["images"]),
            jobs=jobs,
            require_actions=os.environ.get("GITHUB_ACTIONS") == "true",
        )
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        plan = plan if isinstance(plan, dict) else {}
        required = plan.get("required_verification_ids", [])
        deferred = plan.get("deferred_verifications", [])
        deferred_images = plan.get("deferred_images", [])
        verdict = GateVerdict(
            tuple(required) if valid_inventory(required) else (),
            (str(exc),),
            tuple(deferred) if valid_inventory(deferred) else (),
            tuple(deferred_images) if valid_inventory(deferred_images) else (),
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(gate_report(plan, verdict), indent=2) + "\n")
    if path := os.environ.get("GITHUB_OUTPUT"):
        outputs = {
            "status": verdict.status,
            "passed": str(verdict.passed).lower(),
            "required_count": str(len(verdict.required)),
            "deferred_verification_count": str(len(verdict.deferred_verifications)),
            "deferred_image_count": str(len(verdict.deferred_images)),
        }
        with Path(path).open("a") as stream:
            stream.writelines(f"{key}={value}\n" for key, value in outputs.items())
    summary = render_summary(verdict)
    print(summary, end="")
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(path).open("a") as stream:
            stream.write(summary)
    return (
        0
        if verdict.passed
        or (args.allow_deferred and verdict.status == "awaiting-approval")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
