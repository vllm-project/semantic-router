#!/usr/bin/env python3
"""Choose a main push's comparison base when newer pushes cancel older runs.

A newer push to main cancels older push runs that are still qualifying, so a
cancelled commit's changes would go unchecked if each run compared only with
its pushed `before`. A run compares with the newest earlier main commit whose
CI Gate finished, passed or failed: the pushed `before` when nothing was
cancelled, else the last commit whose checks completed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

GATE = "CI / Gate"
FINISHED = {"success", "failure"}
# Beyond this many superseded commits, compare with `before` and say so.
MAX_COMMITS = 50


def gh(path: str) -> dict:
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def first_parents(head: str) -> list[str]:
    """Main's earlier commits, newest first."""
    return subprocess.check_output(
        ["git", "rev-list", "--first-parent", f"--max-count={MAX_COMMITS}", f"{head}^"],
        text=True,
    ).split()


def checked(repository: str, run_ids: list[int]) -> bool:
    """Whether one of a commit's Main runs finished its CI Gate."""
    for run_id in run_ids:
        jobs = gh(f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
        if any(
            job["name"] == GATE and job.get("conclusion") in FINISHED for job in jobs
        ):
            return True
    return False


def comparison_base(repository: str, head: str, before: str) -> str:
    runs = gh(
        f"repos/{repository}/actions/workflows/main.yml/runs?branch=main&per_page=100"
    )["workflow_runs"]
    by_commit: dict[str, list[int]] = {}
    for run in runs:
        by_commit.setdefault(run["head_sha"], []).append(run["id"])
    for sha in first_parents(head):
        if checked(repository, by_commit.get(sha, [])):
            return sha
    print(
        f"::warning::No Main run finished its checks in the {MAX_COMMITS} commits "
        f"before {head}; comparing with the pushed before commit.",
        file=sys.stderr,
    )
    return before


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args()
    try:
        base = comparison_base(args.repository, args.head, args.before)
    except (subprocess.CalledProcessError, KeyError, ValueError) as error:
        print(
            f"::warning::Cannot read earlier Main runs ({error}); comparing with "
            "the pushed before commit, so a cancelled run's changes may go unchecked.",
            file=sys.stderr,
        )
        base = args.before
    # The workflow reads the base from stdout; warnings go to stderr.
    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
