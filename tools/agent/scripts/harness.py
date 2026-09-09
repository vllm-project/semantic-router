#!/usr/bin/env python3
"""Report change impact and run the repository's deterministic checks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CI_DIR = REPO_ROOT / "tools" / "ci"
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

from changed_files import get_changed_files, resolve_base_ref  # noqa: E402
from check_support import (  # noqa: E402
    append_missing_make_target,
    collect_make_targets,
    run_go_lint,
    run_precommit,
    run_python_lint,
    run_reference_config_lint,
    run_rust_lint,
    run_test_commands,
)
from classify_pr_changes import classify  # noqa: E402
from domain_registry import (  # noqa: E402
    commands_for_domains,
    domain_records,
    job_records,
    load_domain_registry,
    profile_records,
    registry_schema_errors,
)


def split_names(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    names = [part.strip() for chunk in raw.split(",") for part in chunk.split()]
    return tuple(dict.fromkeys(name for name in names if name))


def environment_facts(requested_environment: str) -> dict[str, Any]:
    tools = {
        name: bool(shutil.which(name))
        for name in (
            "python3",
            "go",
            "cargo",
            "node",
            "docker",
            "podman",
            "rocminfo",
            "nvidia-smi",
        )
    }
    return {"requested": requested_environment, "tools": tools}


def build_impact(changed_files: list[str], environment: str) -> dict[str, Any]:
    result = classify(changed_files)
    domains = domain_records()
    jobs = job_records()
    return {
        "changed_files": changed_files,
        "test_only": result.test_only,
        "domains": [
            {"name": name, "owner": domains[name]["owner"]} for name in result.domains
        ],
        "checks": list(commands_for_domains(result.domains, "checks")),
        "candidate_ci_jobs": [
            {"name": name, "workflow": jobs[name]["workflow"]}
            for name in result.selected_jobs
        ],
        "optional_e2e_profiles": list(result.profiles),
        "pr_images": list(result.pr_images),
        "environment": environment_facts(environment),
    }


def impact_summary(impact: dict[str, Any]) -> str:
    lines = [f"Changed files ({len(impact['changed_files'])}):"]
    lines.extend(f"  - {path}" for path in impact["changed_files"])
    if not impact["changed_files"]:
        lines.append("  - none")

    lines.append("Domains:")
    lines.extend(
        f"  - {domain['name']} (owner: {domain['owner']})"
        for domain in impact["domains"]
    )
    if not impact["domains"]:
        lines.append("  - none")

    lines.append("Related check commands:")
    lines.extend(f"  - {command}" for command in impact["checks"])
    if not impact["checks"]:
        lines.append("  - none")

    lines.append("Candidate CI jobs:")
    lines.extend(
        f"  - {job['name']}: {job['workflow']}" for job in impact["candidate_ci_jobs"]
    )

    lines.append("Optional E2E profiles:")
    lines.extend(f"  - {profile}" for profile in impact["optional_e2e_profiles"])
    if not impact["optional_e2e_profiles"]:
        lines.append("  - none")

    if impact["pr_images"]:
        lines.append("PR images:")
        lines.extend(f"  - {image}" for image in impact["pr_images"])

    environment = impact["environment"]
    available = [name for name, present in environment["tools"].items() if present]
    missing = [name for name, present in environment["tools"].items() if not present]
    lines.append(f"Environment requested: {environment['requested']}")
    lines.append(f"Host tools available: {', '.join(available) or 'none'}")
    lines.append(f"Host tools absent: {', '.join(missing) or 'none'}")
    return "\n".join(lines)


def validate_harness() -> int:
    registry = load_domain_registry()
    errors = registry_schema_errors(registry)
    make_targets = collect_make_targets()

    for name, job in job_records(registry).items():
        workflow = REPO_ROOT / job["workflow"]
        if not workflow.is_file():
            errors.append(
                f"CI job {name!r} references missing workflow {job['workflow']!r}"
            )

    for name, domain in domain_records(registry).items():
        for field in ("checks", "verify"):
            for command in domain.get(field, []):
                append_missing_make_target(
                    errors, f"domain {name!r} {field}", command, make_targets
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Harness registry is valid.")
    return 0


def run_verify(domains: tuple[str, ...], profiles: tuple[str, ...]) -> int:
    known_domains = domain_records()
    known_profiles = profile_records()
    unknown_domains = sorted(set(domains) - set(known_domains))
    unknown_profiles = sorted(set(profiles) - set(known_profiles))
    if unknown_domains or unknown_profiles:
        if unknown_domains:
            print(f"Unknown domain(s): {', '.join(unknown_domains)}", file=sys.stderr)
        if unknown_profiles:
            print(f"Unknown profile(s): {', '.join(unknown_profiles)}", file=sys.stderr)
        return 2
    if not domains and not profiles:
        print("Specify DOMAIN and/or PROFILE for explicit integration checks.")
        return 2

    unconfigured = [name for name in domains if not known_domains[name].get("verify")]
    if unconfigured:
        print(
            "No integration command configured for: "
            + ", ".join(unconfigured)
            + ". Select a configured DOMAIN or an E2E PROFILE.",
            file=sys.stderr,
        )
        return 2

    commands = list(commands_for_domains(domains, "verify"))
    commands.extend(
        f"make e2e-test E2E_PROFILE={profile} E2E_VERBOSE=true" for profile in profiles
    )
    return run_test_commands(list(dict.fromkeys(commands)), "verification")


def run_check(changed_files: list[str], base_ref: str | None) -> int:
    if not changed_files:
        print("No changed files detected.")
        return 0

    existing = [path for path in changed_files if (REPO_ROOT / path).exists()]
    bootstrap = []
    if any(path.endswith(".go") for path in existing):
        bootstrap.append("make harness-go-bootstrap")
    if any(path.endswith(".rs") for path in existing):
        bootstrap.append("make harness-rust-bootstrap")

    try:
        run_test_commands(bootstrap, "lint tooling")
        for check in (
            lambda: run_precommit(changed_files, base_ref),
            lambda: run_python_lint(changed_files),
            lambda: run_go_lint(changed_files, base_ref),
            lambda: run_reference_config_lint(changed_files),
            lambda: run_rust_lint(changed_files),
        ):
            if (returncode := check()) != 0:
                return returncode
        return 0
    except subprocess.CalledProcessError as exc:
        return exc.returncode


def add_changed_file_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-ref")
    parser.add_argument("--changed-files")
    parser.add_argument("--changed-files-path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    changed = subparsers.add_parser("changed-files")
    add_changed_file_args(changed)

    impact = subparsers.add_parser("impact")
    add_changed_file_args(impact)
    impact.add_argument("--env", default="cpu", choices=("cpu", "amd", "nvidia"))
    impact.add_argument("--format", default="summary", choices=("summary", "json"))

    checks = subparsers.add_parser("check")
    add_changed_file_args(checks)

    formatting = subparsers.add_parser("format")
    add_changed_file_args(formatting)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--domains")
    verify.add_argument("--profiles")

    subparsers.add_parser("validate")
    return parser


def changed_files_for_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[str]:
    try:
        return get_changed_files(
            getattr(args, "changed_files", None),
            getattr(args, "base_ref", None),
            getattr(args, "changed_files_path", None),
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable") from exc


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        return validate_harness()
    if args.command == "verify":
        return run_verify(split_names(args.domains), split_names(args.profiles))

    try:
        base_ref = resolve_base_ref(getattr(args, "base_ref", None))
    except ValueError as exc:
        parser.error(str(exc))
    changed_files = changed_files_for_args(parser, args)
    if args.command == "changed-files":
        print("\n".join(changed_files))
        return 0
    if args.command == "impact":
        impact = build_impact(changed_files, args.env)
        if args.format == "json":
            print(json.dumps(impact, indent=2))
        else:
            print(impact_summary(impact))
        return 0
    if args.command == "check":
        return run_check(changed_files, base_ref)
    if args.command == "format":
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "precommit_tool.py"),
                "format",
                *changed_files,
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
