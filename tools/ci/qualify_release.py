#!/usr/bin/env python3
"""Require successful Main validation for the exact commit being released."""

from __future__ import annotations

import argparse
import json
import os
import subprocess


def qualified_run(runs: list[dict], sha: str) -> dict:
    matching = [
        run
        for run in runs
        if run.get("head_sha") == sha
        and run.get("event") == "push"
        and run.get("head_branch") == "main"
    ]
    if not matching:
        raise ValueError(
            f"No Main push run exists for release commit {sha}; "
            "publish a commit already validated on main"
        )
    latest = max(matching, key=lambda run: (run["id"], run.get("run_attempt", 1)))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError(
            f"Main must succeed for {sha}: {latest.get('html_url', latest['id'])} "
            f"is {latest.get('status')}/{latest.get('conclusion')}"
        )
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/workflows/main.yml/runs?head_sha={args.sha}&event=push&per_page=100",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        run = qualified_run(json.loads(result.stdout)["workflow_runs"], args.sha)
    except ValueError as exc:
        print(
            f"::error::{exc}. Complete or rerun Main for this exact commit, "
            "then rerun Release. workflow_dispatch only checks version contracts."
        )
        return 1
    print(f"Qualified release commit {args.sha}: {run['html_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
