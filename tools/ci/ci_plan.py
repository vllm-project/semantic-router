#!/usr/bin/env python3
"""Produce one immutable verification and build plan for every CI entrypoint."""
from __future__ import annotations

import argparse
import hashlib
import json
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
            reasons.append("full-cpu-v1")
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


def github_outputs(plan: dict) -> dict[str, str]:
    # Only labels enter the Actions matrix; full records stay outside it so
    # GitHub cannot append their fields to a static caller name.
    dispatch = {"tools": records_by_display_name(plan["component_batches"])}
    outputs = {
        "plan": json.dumps(plan, separators=(",", ":")),
        "images": json.dumps(plan["images"]),
        "publish_images": json.dumps(plan["publish_images"]),
        "build_native": str(plan["native"]).lower(),
        "multiarch": str(plan["multiarch"]).lower(),
        "publish_helm": str(plan["publish_helm"]).lower(),
        "publish_python": str(plan["publish_python"]).lower(),
        "component_batches": json.dumps(
            plan["component_batches"], separators=(",", ":")
        ),
    }
    for executor in EXECUTORS:
        if executor == "tools":
            continue
        records = [r for r in plan["verifications"] if r["executor"] == executor]
        outputs[executor] = json.dumps(records, separators=(",", ":"))
        if executor not in {"quality", "generated"}:
            dispatch[executor] = records_by_display_name(records)
    outputs["dispatch"] = json.dumps(dispatch, separators=(",", ":"))
    return outputs


def records_by_display_name(records: list[dict]) -> dict[str, dict]:
    indexed = {}
    for record in records:
        label = record.get("display_name")
        if not isinstance(label, str) or not label.strip() or label in indexed:
            raise ValueError("CI matrix display names must be nonempty and unique")
        indexed[label] = record
    return indexed


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    if args.github_output:
        with args.github_output.open("a") as stream:
            for key, value in github_outputs(plan).items():
                stream.write(f"{key}={value}\n")
    print(
        f"Planned {len(plan['verifications'])} verifications; {len(plan['images'])} image builds; full CPU={plan['full_cpu']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
