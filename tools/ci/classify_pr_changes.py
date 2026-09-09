#!/usr/bin/env python3
"""Map changed paths to coarse CI domains and explicit risk escalations."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from domain_registry import (
    any_matches,
    domain_records,
    image_records,
    job_records,
    matching_domains,
    profile_records,
)

PRODUCTION_RELEASE_IMAGES = (
    "dashboard",
    "extproc",
    "extproc-rocm",
    "operator",
    "operator-bundle",
    "vllm-sr",
    "vllm-sr-cuda",
    "vllm-sr-rocm",
)
NIGHTLY_IMAGES = (
    "anthropic-shim",
    *PRODUCTION_RELEASE_IMAGES,
    "llm-katan",
    "vllm-sr-sim",
)


@dataclass(frozen=True)
class Classification:
    signals: dict[str, bool]
    domains: tuple[str, ...]
    profiles: tuple[str, ...]
    pr_images: tuple[str, ...]
    publish_images: tuple[str, ...]
    selected_jobs: tuple[str, ...]
    test_only: bool

    def github_outputs(self) -> dict[str, str]:
        outputs = {name: str(value).lower() for name, value in self.signals.items()}
        outputs.update(
            {
                "domains": _compact_json(self.domains),
                "e2e": str(bool(self.profiles)).lower(),
                "e2e_profiles": _compact_json(self.profiles),
                "full_e2e_profiles": _compact_json(full_e2e_profiles()),
                "images": str(bool(self.pr_images)).lower(),
                "pr_images": _compact_json(self.pr_images),
                "publish_images": _compact_json(self.publish_images),
            }
        )
        return outputs


def _compact_json(values: tuple[str, ...]) -> str:
    return json.dumps(values, separators=(",", ":"))


def is_documentation_path(path: str) -> bool:
    return (
        path.startswith("website/")
        or path.endswith((".md", ".mdx"))
        or is_repository_ownership_path(path)
    )


def is_repository_ownership_path(path: str) -> bool:
    return path == ".github/CODEOWNERS" or Path(path).name == "OWNER"


def is_test_path(path: str) -> bool:
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    return (
        name.startswith("test_")
        or name.endswith(("_test.go", "_test.py", "_test.rs"))
        or ".test." in name
        or ".spec." in name
        or "tests" in parts
        or "testcases" in parts
        or "testdata" in parts
    )


def executable_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if not is_documentation_path(path))


def is_test_only_change(paths: tuple[str, ...]) -> bool:
    executable = executable_paths(paths)
    return bool(executable) and all(is_test_path(path) for path in executable)


def full_e2e_profiles() -> tuple[str, ...]:
    return tuple(
        name for name, data in profile_records().items() if data.get("full_ci")
    )


def select_profiles(
    changed: tuple[str, ...], *, full: bool, suppress_expensive: bool
) -> tuple[str, ...]:
    if full:
        return full_e2e_profiles()
    if suppress_expensive:
        return ()
    return tuple(
        name
        for name, data in profile_records(selection="pr").items()
        if any_matches(changed, data.get("paths", []))
    )


def select_images(
    changed: tuple[str, ...], *, field: str, suppress_expensive: bool = False
) -> tuple[str, ...]:
    if suppress_expensive:
        return ()
    return tuple(
        name
        for name, data in image_records().items()
        if any_matches(changed, data.get(field, []))
    )


def select_jobs(
    changed: tuple[str, ...],
    domains: tuple[str, ...],
    profiles: tuple[str, ...],
    pr_images: tuple[str, ...],
    *,
    full: bool,
    docs_only: bool,
    suppress_expensive: bool,
) -> tuple[str, ...]:
    jobs = job_records()
    enabled = {name for name, data in jobs.items() if data.get("always")}
    if full:
        enabled.update(name for name, data in jobs.items() if data.get("full"))
    if not docs_only:
        enabled.add("security")

    records = domain_records()
    for domain_name in domains:
        domain = records[domain_name]
        enabled.update(domain.get("ci_jobs", []))
    # Escalation paths are independent of ownership paths (for example shared
    # Docker tooling can require CLI integration without belonging to the CLI).
    for domain in records.values():
        escalation = domain.get("escalation", {})
        if not suppress_expensive and any_matches(changed, escalation.get("paths", [])):
            enabled.update(escalation.get("jobs", []))

    # A reusable workflow edit runs that workflow's own job, but does not imply
    # unrelated product or deployment coverage.
    for name, data in jobs.items():
        if data.get("workflow") in changed:
            enabled.add(name)
    if profiles:
        enabled.add("e2e")
    if pr_images:
        enabled.add("images")
    return tuple(name for name in jobs if name in enabled)


def build_signals(
    selected_jobs: tuple[str, ...],
    changed: tuple[str, ...],
    domains: tuple[str, ...],
    *,
    full: bool,
    docs_only: bool,
) -> dict[str, bool]:
    selected = set(selected_jobs)
    signals = {
        str(data["output"]): name in selected
        for name, data in job_records().items()
        if data.get("output")
    }
    signals.update(
        {
            "website": any(path.startswith("website/") for path in changed),
            "docs_only": docs_only,
            "helm": "helm" in domains,
            "full": full,
        }
    )
    return signals


def classify(
    paths: list[str] | tuple[str, ...], *, full: bool = False
) -> Classification:
    changed = tuple(sorted(set(paths)))
    executable = executable_paths(changed)
    docs_only = bool(changed) and not executable
    records = domain_records()
    executable_domains = set(
        matching_domains(
            path for path in executable if not is_repository_ownership_path(path)
        )
    )
    documentation_domains = {
        name
        for name in matching_domains(changed)
        if records[name].get("documentation_check")
    }
    domains = tuple(
        name
        for name in records
        if name in executable_domains or name in documentation_domains
    )
    test_only = is_test_only_change(changed)
    # Unit tests under source trees stay in their owning domain. E2E files are
    # already explicit deployment contracts and may select their named profile.
    suppress_expensive = docs_only or (
        test_only and not any(path.startswith("e2e/") for path in changed)
    )
    profiles = select_profiles(
        changed, full=full, suppress_expensive=suppress_expensive
    )
    pr_images = select_images(
        changed, field="pr_paths", suppress_expensive=suppress_expensive
    )
    publish_images = select_images(
        changed, field="publish_paths", suppress_expensive=suppress_expensive
    )
    selected_jobs = select_jobs(
        changed,
        domains,
        profiles,
        pr_images,
        full=full,
        docs_only=docs_only,
        suppress_expensive=suppress_expensive,
    )
    return Classification(
        signals=build_signals(
            selected_jobs,
            changed,
            domains,
            full=full,
            docs_only=docs_only,
        ),
        domains=domains,
        profiles=profiles,
        pr_images=pr_images,
        publish_images=publish_images,
        selected_jobs=selected_jobs,
        test_only=test_only,
    )


def git_changed_files(base: str, head: str) -> list[str]:
    if base and set(base) != {"0"}:
        result = subprocess.run(
            ["git", "diff", "--name-only", "-z", base, head],
            check=True,
            capture_output=True,
        )
    else:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    return [path.decode() for path in result.stdout.split(b"\0") if path]


def write_github_outputs(path: Path, classification: Classification) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in classification.github_outputs().items():
            output.write(f"{name}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("paths", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or git_changed_files(args.base, args.head)
    result = classify(paths, full=args.full)
    if args.github_output:
        write_github_outputs(args.github_output, result)
    else:
        print(
            json.dumps(
                {
                    "paths": sorted(paths),
                    "domains": result.domains,
                    "test_only": result.test_only,
                    "jobs": result.selected_jobs,
                    "profiles": result.profiles,
                    "pr_images": result.pr_images,
                    "publish_images": result.publish_images,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
