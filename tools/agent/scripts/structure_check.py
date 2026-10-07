#!/usr/bin/env python3
"""Check changed files against dependency rules and root-file placement."""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = REPO_ROOT / "tools" / "agent" / "structure-rules.yaml"


@dataclass
class Finding:
    level: str
    file: str
    message: str


def load_rules() -> dict:
    with RULES_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def should_ignore(path: str, rules: dict) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in rules["ignore_globs"])


def load_baseline_source(path: str, base_ref: str | None) -> str | None:
    ref = base_ref or "HEAD"
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def evaluate_dependency_rules(
    path: str, text: str, rules: dict, base_ref: str | None = None
) -> list[Finding]:
    findings: list[Finding] = []
    baseline_source: str | None = None
    for rule in rules["dependency_rules"]:
        if not any(fnmatch.fnmatch(path, pattern) for pattern in rule["applies_to"]):
            continue
        for literal in rule["forbidden_literals"]:
            current_count = text.count(literal)
            if current_count == 0:
                continue
            if rule.get("policy", "error") == "no-new":
                if baseline_source is None:
                    baseline_source = load_baseline_source(path, base_ref) or ""
                baseline_count = baseline_source.count(literal)
                if current_count <= baseline_count:
                    findings.append(
                        Finding(
                            level="WARN",
                            file=path,
                            message=(
                                f"{rule['name']}: pre-existing forbidden dependency "
                                f"'{literal}' did not grow from baseline {baseline_count}"
                            ),
                        )
                    )
                    continue
            findings.append(
                Finding(
                    level="ERROR",
                    file=path,
                    message=f"{rule['name']}: forbidden dependency '{literal}'",
                )
            )
    return findings


def evaluate_root_placement(path: str, rules: dict) -> list[Finding]:
    """Reject new root files that do not have a repository-wide contract."""
    absolute_path = REPO_ROOT / path
    if "/" in path or not absolute_path.is_file():
        return []
    if path in rules["root_files"]["allowed"]:
        return []
    return [
        Finding(
            level="ERROR",
            file=path,
            message="root file is not allowlisted; place it under its owning subtree",
        )
    ]


def evaluate_file(
    path: str, rules: dict[str, Any], base_ref: str | None
) -> list[Finding]:
    governed = any(
        fnmatch.fnmatch(path, pattern)
        for rule in rules["dependency_rules"]
        for pattern in rule["applies_to"]
    )
    if not governed or should_ignore(path, rules):
        return []

    absolute_path = REPO_ROOT / path
    if not absolute_path.exists():
        return []

    source_text = absolute_path.read_bytes().decode("utf-8", errors="ignore")
    return evaluate_dependency_rules(path, source_text, rules, base_ref)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run structure checks on changed files"
    )
    parser.add_argument("files", nargs="*")
    parser.add_argument("--base-ref", default=os.getenv("BASE_REF"))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    rules = load_rules()
    findings_by_file: dict[str, list[Finding]] = defaultdict(list)

    for raw_path in args.files:
        path = raw_path.strip()
        while path.startswith("./"):
            path = path[2:]
        if not path:
            continue
        for finding in evaluate_root_placement(path, rules):
            findings_by_file[finding.file].append(finding)
        for finding in evaluate_file(path, rules, args.base_ref):
            findings_by_file[finding.file].append(finding)

    exit_code = 0
    for file_path in sorted(findings_by_file):
        for finding in findings_by_file[file_path]:
            if finding.level == "ERROR":
                exit_code = 1
            print(f"[{finding.level}] {finding.file} :: {finding.message}")

    if not findings_by_file:
        print("Structure check passed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
