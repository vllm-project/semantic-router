#!/usr/bin/env python3
"""Produce one immutable verification and build plan for every CI entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from classify_pr_changes import (
    NIGHTLY_IMAGES,
    PRODUCTION_RELEASE_IMAGES,
    classify,
    git_changed_files,
)
from domain_registry import domain_records, load_domain_registry, matching_domains
from execution_batches import (
    EXECUTOR_JOBS,
    dispatch_job,
    e2e_batches,
    expected_dispatch_jobs,
    image_producers,
    native_batches,
)
from provider_mocker_image import (
    IMAGE as MOCKER_IMAGE,
)
from provider_mocker_image import (
    PublicationMissingError,
    acquisition,
    published_from_plan,
    resolve_published,
)
from verification_catalog import (
    catalog_errors,
    full_cpu_ids,
    load_catalog,
    verification_records,
)

PROFILES = ("pr", "main", "nightly", "release")
GIT_SHA_LENGTH = 40
EXECUTORS = (
    "quality",
    "generated",
    "security",
    "core",
    "storage",
    "dashboard",
    "operator",
    "local",
    "recipes",
    "native",
    "performance",
    "package",
    "tools",
    "e2e",
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_image_sources(plan: dict) -> None:
    """Reuse exact-input fixtures when available, otherwise qualify them here."""
    published = published_from_plan(plan)
    if not published:
        return
    try:
        plan["image_sources"][MOCKER_IMAGE] = resolve_published(published)
    except PublicationMissingError:
        if plan["profile"] not in {"pr", "main"}:
            # Release and nightly runs require an already qualified main image.
            raise
        # A failed main gate can leave a valid fixture build unpublished. PRs
        # still qualify their own exact inputs, without registry write access.
        published["source"] = "candidate"
        plan["build_images"] = sorted({*plan["build_images"], MOCKER_IMAGE})
        if plan["profile"] == "main":
            # The next successful main gate promotes this sealed candidate.
            plan["publish_images"] = sorted({*plan["publish_images"], MOCKER_IMAGE})
            plan["multiarch"] = True
    plan["plan_sha256"] = digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def make_plan(
    paths: list[str],
    *,
    source_sha: str,
    profile: str = "pr",
    full: bool = False,
    draft: bool = False,
    base_sha: str = "",
    requested: tuple[str, ...] = (),
) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown CI profile: {profile}")
    if len(source_sha) != GIT_SHA_LENGTH or any(
        c not in "0123456789abcdef" for c in source_sha
    ):
        raise ValueError("plan requires a full lowercase source SHA")
    registry = load_domain_registry()
    errors = catalog_errors(registry)
    if errors:
        raise ValueError("; ".join(errors))
    full = full or profile in {"nightly", "release"}
    selection = classify(paths, full=full)
    records = verification_records(registry)
    ids = [] if draft and profile == "pr" else list(selection.selected_jobs)
    if requested:
        if profile != "pr" or full or draft or paths:
            raise ValueError(
                "explicit verification selection requires an isolated manual plan"
            )
        if len(requested) != len(set(requested)) or set(requested) - set(records):
            raise ValueError("explicit verification inventory is unknown or duplicated")
        ids = list(requested)
    verifications = []
    for name in ids:
        record = {"id": name, **records[name], "source_sha": source_sha}
        reasons = [
            f"domain:{domain}"
            for domain in matching_domains(paths)
            if name in domain_records()[domain].get("verifications", [])
            or name
            in domain_records()[domain].get("escalation", {}).get("verifications", [])
        ]
        if full and name in full_cpu_ids():
            reasons.append(f"full-cpu-v{load_catalog()['full_cpu']['version']}")
        record["reasons"] = (
            ["manual-selection"]
            if requested
            else reasons or ["always" if record.get("always") else "explicit-consumer"]
        )
        if record["executor"] == "operator":
            variants = (
                (
                    "memory",
                    "redis",
                    "milvus",
                    "qdrant",
                    "hybrid",
                    "mmbert",
                    "complexity",
                )
                if full
                else ("memory",)
            )
            record["integration_matrix"] = [
                {
                    "cache-backend": backend,
                    "deploy-redis": backend == "redis",
                    "deploy-milvus": backend in {"milvus", "hybrid"},
                    "deploy-qdrant": backend == "qdrant",
                    "router-name": f"test-router-{backend}",
                }
                for backend in variants
            ]
        if record["executor"] == "e2e":
            record["baseline_suite"] = "full" if full else "standard"
        record["dispatch_job"] = dispatch_job(record)
        record["contract_sha256"] = digest(record)
        verifications.append(record)
    publish_images = []
    if profile == "main":
        publish_images = list(selection.publish_images)
    elif profile == "nightly":
        publish_images = list(NIGHTLY_IMAGES)
    elif profile == "release":
        publish_images = list(PRODUCTION_RELEASE_IMAGES)
    images = sorted(
        set(selection.pr_images if ids else ())
        | set(publish_images)
        | {image for record in verifications for image in record["images"]}
    )
    image_sources = (
        {MOCKER_IMAGE: acquisition(paths if profile in {"pr", "main"} else [])}
        if MOCKER_IMAGE in images
        else {}
    )
    reused = [
        name
        for name, record in image_sources.items()
        if record["source"] == "published"
    ]
    build_images = [name for name in images if name not in reused]
    plan = {
        "schema_version": 1,
        "source_sha": source_sha,
        "base_sha": base_sha,
        "profile": profile,
        "full_cpu": full,
        "full_cpu_version": load_catalog()["full_cpu"]["version"],
        "draft": draft,
        "paths": sorted(set(paths)),
        "requested_verifications": list(requested),
        "domains": list(selection.domains),
        "verifications": verifications,
        "expected_verification_ids": ids,
        "images": images,
        "build_images": build_images,
        "image_sources": image_sources,
        "native": any(record["native"] for record in verifications),
        "publish_images": publish_images,
        "publish_helm": profile in {"nightly", "release"}
        or (profile == "main" and selection.signals["helm"]),
        "publish_python": profile == "release"
        or (
            profile == "main"
            and bool(
                {"vllm-sr-cli", "generated-model-catalog"} & set(selection.domains)
            )
        ),
        "multiarch": bool(publish_images),
        "not_applicable": load_catalog()["full_cpu"]["excluded"],
        "quality_context": dict(selection.signals),
    }
    plan["component_batches"] = component_batches(verifications)
    plan["native_batches"] = native_batches(verifications)
    plan["e2e_batches"] = e2e_batches(verifications)
    plan["image_producers"] = image_producers(images)
    plan["expected_dispatch_jobs"] = expected_dispatch_jobs(plan)
    plan["plan_sha256"] = digest(plan)
    return plan


def component_batches(verifications: list[dict]) -> list[dict]:
    """Pack compatible lightweight contracts without changing their identities."""
    return [
        {"id": name, **worker, "verifications": selected}
        for name, worker in load_catalog()["component_workers"].items()
        if (
            selected := [
                row
                for row in verifications
                if row["executor"] == "tools" and row["worker"] == name
            ]
        )
    ]


def dispatch_records(plan: dict) -> dict[str, list[dict]]:
    """Map stable physical caller IDs to contracts or bounded workers."""
    result = {job: [] for job in EXECUTOR_JOBS}
    for record in plan["verifications"]:
        if record["executor"] not in {"tools", "native", "e2e"}:
            result[record["dispatch_job"]].append(record)
    result["tools"] = plan["component_batches"]
    for batch in [*plan["native_batches"], *plan["e2e_batches"]]:
        result[batch["dispatch_job"]].append(batch)
    return result


def display_dispatch(plan: dict) -> dict[str, dict]:
    """Only scalar worker labels enter Actions matrices."""
    records = dispatch_records(plan)
    dispatch = {
        job: records_by_display_name(rows)
        for job, rows in records.items()
        if job != "local"
    }
    dispatch["local"] = {
        f"Shard {index}": row
        for index, row in enumerate(
            sorted(records["local"], key=lambda row: row["id"]), 1
        )
    }
    return dispatch


def github_outputs(plan: dict) -> dict[str, str]:
    records = dispatch_records(plan)
    dispatch = display_dispatch(plan)
    published = published_from_plan(plan)
    producers = {
        job: {
            "images": selected,
            "build_images": [
                image for image in selected if image in plan["build_images"]
            ],
            "published_images": (
                [published] if published and published["id"] in selected else []
            ),
        }
        for job, selected in plan["image_producers"].items()
    }
    values = {
        # The complete plan is uploaded as ci-plan for the gate. Passing it as a
        # job output exceeds GitHub's 1 MiB UTF-16 limit for release profiles;
        # callers only need these fields to select their workflow behavior.
        "plan": {
            "profile": plan["profile"],
            "quality_context": plan["quality_context"],
        },
        "dispatch": dispatch,
        "worker_labels": {job: list(rows) for job, rows in dispatch.items()},
        "image_producers": producers,
        "images": plan["images"],
        "build_images": plan["build_images"],
        "published_images": [published] if published else [],
        "publish_images": plan["publish_images"],
        "build_native": plan["native"],
        "multiarch": plan["multiarch"],
        "publish_helm": plan["publish_helm"],
        "publish_python": plan["publish_python"],
        "component_batches": plan["component_batches"],
    }
    values.update(records)
    return {
        key: json.dumps(value, separators=(",", ":")) for key, value in values.items()
    }


def records_by_display_name(records: list[dict]) -> dict[str, dict]:
    indexed = {}
    for record in records:
        label = record.get("display_name")
        if not isinstance(label, str) or not label.strip() or label in indexed:
            raise ValueError("CI matrix display names must be nonempty and unique")
        indexed[label] = record
    return indexed


def render_plan_summary(plan: dict) -> str:
    paths = {
        "quality": "Quality",
        "components": "Tests / Components",
        "integration": "Tests / Integration",
        "conformance": "Tests / Conformance",
        "runtime": "Tests / Runtime",
        "e2e": "Tests / E2E",
        "performance": "Tests / Performance",
        "packages": "Tests / Packages",
    }
    workers = {}
    for job, rows in display_dispatch(plan).items():
        for label, row in rows.items():
            for record in row.get("verifications", [row]):
                workers[record["id"]] = job + " / " + label
    lines = [
        "| Category | Planned contract | Worker | Runtime / device / platform |",
        "| --- | --- | --- | --- |",
    ]
    for record in sorted(
        plan["verifications"], key=lambda row: (row["category"], row["id"])
    ):
        lines.append(
            f"| {paths[record['category']]} | {record['display_name']} | "
            f"{workers[record['id']]} | {record['runtime']} / {record['device']} / {record['platform']} |"
        )
    return "\n".join(lines) + "\n"


def previous_release(version: str, tags: list[str]) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", version)
    if not match:
        raise ValueError("release version is not semantic version syntax")
    current = tuple(map(int, match.groups()))
    candidates = []
    for tag in tags:
        parsed = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
        if parsed:
            value = tuple(map(int, parsed.groups()))
            if value[0] == current[0] and value < current:
                candidates.append((value, tag))
    if not candidates:
        raise ValueError(
            "no compatible previous stable release for paired performance; declare a supported comparison baseline before release qualification"
        )
    return max(candidates)[1]


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "previous-release":
        parser = argparse.ArgumentParser(
            description="Resolve an ancestor stable release in the same major version"
        )
        parser.add_argument("--version", required=True)
        parser.add_argument("--github-output", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        tags = subprocess.check_output(
            ["git", "tag", "--merged", "HEAD"], text=True
        ).splitlines()
        try:
            ref = previous_release(args.version, tags)
        except ValueError as exc:
            parser.exit(1, str(exc) + "\n")
        with args.github_output.open("a") as stream:
            stream.write(f"ref={ref}\n")
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="pr")
    parser.add_argument("--base", default="")
    parser.add_argument("--head", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--verification", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    plan = make_plan(
        (
            args.paths
            if args.verification
            else args.paths or git_changed_files(args.base, args.head)
        ),
        source_sha=args.head,
        base_sha=args.base,
        profile=args.profile,
        full=args.full,
        draft=args.draft,
        requested=tuple(args.verification),
    )
    resolve_image_sources(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    if args.github_output:
        with args.github_output.open("a") as stream:
            for key, value in github_outputs(plan).items():
                stream.write(f"{key}={value}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write(render_plan_summary(plan))
    print(
        f"Planned {len(plan['verifications'])} verifications; {len(plan['build_images'])} image builds; full CPU={plan['full_cpu']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
