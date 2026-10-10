#!/usr/bin/env python3
"""Report failure and retry rates per e2e case over recent merged and PR runs.

A single run cannot tell a flake from a regression. This reads the artifacts of
several runs and reports, per case, how often it failed and how often it needed
a retry.

`ci.yml` has no runs of its own. It is only reachable through `workflow_call`, so
its jobs and artifacts belong to the caller's run. The callers are:

- `main.yml` and `nightly-build.yml`, which run merged code. A failure there is a
  flake or a regression on main, never a bug in an unmerged branch, so they are
  the denominator for failures and they also carry retries.
- `pr.yml`, which runs unmerged code. It carries retries, but a failure says more
  about the branch than about the suite, so it counts towards retries only.

The workflow alone does not decide this. `nightly-build.yml` also takes
`workflow_dispatch`, which can be pointed at any branch, so a run whose
`head_branch` is not the merged branch counts towards retries only, whichever
workflow produced it.

Only runs that finished with a result are read. A cancelled run never settles its
e2e jobs, and in `pr.yml` every new push cancels the run before it, so reading
cancelled runs would fill a workflow's sample with runs that reported nothing.

Two artifacts describe the same execution, so both are read:

- `ci-result-*` holds the receipt the CI gate uses. It is uploaded with
  `if-no-files-found: error`, so it proves which profiles ran.
- `e2e-evidence-*` holds the raw framework report, the only place that records
  how many attempts a case needed. It is uploaded with
  `if-no-files-found: warn`, so a run can have a receipt without raw evidence.

Reading both keeps final failures visible when the raw evidence is missing.

Reads only local files when `--from-dir` is given, so the aggregation is testable
without GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

DEFAULT_WORKFLOWS = ("main.yml", "nightly-build.yml")
DEFAULT_RETRY_WORKFLOWS = ("pr.yml",)
DEFAULT_DOWNLOAD_DIR = Path(".agent-harness/flaky/runs")

# The workflow-runs endpoint returns at most 100 runs per page.
FETCH_LIMIT = 100

# A cancelled or skipped run stops before its e2e jobs settle, so it holds no
# result to read. Cancelled runs are the common case in `pr.yml`, where every new
# push supersedes the run before it, so keeping them would fill the sample for a
# whole workflow with runs that never reported anything.
USABLE_CONCLUSIONS = frozenset({"success", "failure"})

# The branch whose runs hold merged code. `main.yml` only runs on a push to
# `main`, but `nightly-build.yml` also takes `workflow_dispatch`, which can be
# pointed at any branch. A run of unmerged code says more about that branch than
# about the suite, so the branch decides the provenance, not the workflow.
MERGED_BRANCH = "main"


@dataclass
class RunRef:
    """One completed workflow run whose artifacts are read."""

    workflow: str
    run_id: str
    created_at: str = ""
    conclusion: str = ""
    head_branch: str = ""
    retry_workflow: bool = False
    note: str = ""

    @property
    def ran_merged_code(self) -> bool:
        """Whether the run holds code that is on the merged branch."""
        return self.head_branch in ("", MERGED_BRANCH)

    @property
    def retry_only(self) -> bool:
        """A run counts towards retries only unless it ran merged code."""
        return self.retry_workflow or not self.ran_merged_code

    @property
    def directory(self) -> str:
        """Top-level download directory, which is also the run's identity."""
        return f"{self.workflow.removesuffix('.yml')}-{self.run_id}"

    def describe(self) -> str:
        day = self.created_at[:10] or "unknown date"
        # Name the branch only where it changes the provenance. `pr.yml` runs
        # are already marked retry-only, so their branch adds nothing.
        branch = (
            f", `{self.head_branch}`"
            if self.retry_workflow is False and not self.ran_merged_code
            else ""
        )
        return f"`{self.run_id}` ({day}, {self.conclusion or 'unknown'}{branch})"


@dataclass
class CaseOutcome:
    """What one run's profile report says about one case."""

    passed: bool
    attempts: int = 1


@dataclass
class Sample:
    """One profile report from one run."""

    run: str
    profile: str
    retry_only: bool = False
    has_evidence: bool = False
    cases: dict[str, CaseOutcome] = field(default_factory=dict)


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        print(f"skipping unreadable artifact {path}: {error}", file=sys.stderr)
        return None


def run_directory_of(path: Path, root: Path) -> str:
    """Return the run a downloaded artifact belongs to.

    Every run is downloaded below its own top-level directory, so the first path
    component identifies it. A flat directory of reports is one run.
    """
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else "."


def load_samples(root: Path, runs: list[RunRef]) -> list[Sample]:
    """Merge the receipts and the raw reports below `root` into one sample per run and profile."""
    known = {run.directory: run for run in runs}
    samples: dict[tuple[str, str], Sample] = {}

    def sample_for(directory: str, profile: str) -> Sample:
        run = known.get(directory)
        return samples.setdefault(
            (directory, profile),
            Sample(
                run=directory,
                profile=profile,
                retry_only=run.retry_only if run else False,
            ),
        )

    for path in sorted(root.glob("**/ci-result-*/*.json")):
        receipt = read_json(path)
        evidence = receipt.get("evidence") if isinstance(receipt, dict) else None
        if not isinstance(evidence, dict) or not isinstance(
            evidence.get("cases"), list
        ):
            continue
        profile = evidence.get("profile")
        if not isinstance(profile, str) or not profile:
            continue
        sample = sample_for(run_directory_of(path, root), profile)
        for case in evidence["cases"]:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str):
                continue
            sample.cases.setdefault(
                case["id"], CaseOutcome(passed=case.get("status") == "passed")
            )

    for path in sorted(root.glob("**/e2e-evidence-*/**/test-report.json")):
        report = read_json(path)
        if not isinstance(report, dict) or not isinstance(
            report.get("test_results"), list
        ):
            continue
        profile = report.get("profile")
        if not isinstance(profile, str) or not profile:
            continue
        sample = sample_for(run_directory_of(path, root), profile)
        sample.has_evidence = True
        for row in report["test_results"]:
            name = row.get("Name") if isinstance(row, dict) else None
            if not isinstance(name, str) or not name:
                continue
            attempts = row.get("Attempts")
            # Reports written before retries existed carry no Attempts.
            attempts = attempts if isinstance(attempts, int) and attempts >= 1 else 1
            # The raw report describes the same execution as the receipt, so it
            # replaces the receipt row instead of adding a second one.
            sample.cases[name] = CaseOutcome(
                passed=row.get("Passed") is True, attempts=attempts
            )

    return list(samples.values())


def aggregate(samples: list[Sample]) -> list[dict]:
    """Return failure and retry rates per case, worst first."""
    merged: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    failures: dict[tuple[str, str], int] = defaultdict(int)
    retries: dict[tuple[str, str], int] = defaultdict(int)

    for sample in samples:
        for name, outcome in sample.cases.items():
            key = (sample.profile, name)
            totals[key] += 1
            # A failure in an unmerged branch says more about the branch than
            # about the suite, so PR runs stay out of the failure rate.
            if not sample.retry_only:
                merged[key] += 1
                if not outcome.passed:
                    failures[key] += 1
            # A case that never passed is a break, not a flake. Counting it as a
            # retry would report a branch that breaks a case as that case being
            # flaky, and its failures already have their own column.
            if outcome.passed and outcome.attempts > 1:
                retries[key] += 1

    rows = []
    for key, count in totals.items():
        profile, name = key
        if not failures[key] and not retries[key]:
            continue
        rows.append(
            {
                "profile": profile,
                "case": name,
                "merged": merged[key],
                "samples": count,
                "failures": failures[key],
                "retries": retries[key],
                "failure_rate": failures[key] / merged[key] if merged[key] else 0.0,
                "retry_rate": retries[key] / count,
            }
        )
    rows.sort(
        key=lambda row: (
            -row["failure_rate"],
            -row["retry_rate"],
            row["profile"],
            row["case"],
        )
    )
    return rows


def render(
    rows: list[dict], samples: list[Sample], runs: list[RunRef], problems: list[str]
) -> str:
    """Render the summary, including what could not be read and why."""
    evidence = sum(1 for sample in samples if sample.has_evidence)
    reports = Counter(sample.run for sample in samples)

    def artifacts(run: RunRef) -> str:
        if reports[run.directory]:
            return f"{reports[run.directory]} profile report(s)"
        return run.note or "no e2e artifacts"

    lines = [
        "## e2e failure and retry rates",
        "",
        f"Read {len(runs)} run(s); {len(samples)} profile report(s), "
        f"{evidence} with raw e2e evidence.",
        "",
    ]
    if runs:
        lines += ["<details><summary>Runs read</summary>", ""]
        for workflow in dict.fromkeys(run.workflow for run in runs):
            group = [run for run in runs if run.workflow == workflow]
            suffix = " (retries only)" if group[0].retry_workflow else ""
            described = ", ".join(
                f"{run.describe()}, {artifacts(run)}" for run in group
            )
            lines.append(f"- `{workflow}`{suffix}: {described}")
        lines += ["", "</details>", ""]
    if problems:
        lines += ["<details><summary>Collection problems</summary>", ""]
        lines += [f"- {problem}" for problem in problems]
        lines += ["", "</details>", ""]
    lines += [
        "A failure is a case that failed every attempt in a merged run. A retry is "
        "a case that passed after more than one attempt in any run.",
        "",
    ]
    if not rows:
        lines.append("No case failed and no case needed a retry in this window.")
        return "\n".join(lines) + "\n"

    lines += [
        "| Profile | Case | Merged runs | Failures | Failure rate "
        "| Runs | Retries | Retry rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['profile']}` | `{row['case']}` | {row['merged']} "
            f"| {row['failures']} | {row['failure_rate']:.1%} "
            f"| {row['samples']} | {row['retries']} | {row['retry_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def repo_slug() -> str:
    """Resolve owner/repo for the workflow-runs API."""
    if slug := os.environ.get("GH_REPO", "").strip():
        return slug
    resolved = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        capture_output=True,
        text=True,
        check=False,
    )
    if resolved.returncode != 0 or not resolved.stdout.strip():
        raise SystemExit(
            "set GH_REPO=<owner>/<repo>, or run inside a clone of the repository"
        )
    return resolved.stdout.strip()


def recent_runs(
    workflow: str, limit: int, days: int, repo: str, *, retry_only: bool
) -> tuple[list[RunRef], str]:
    """Return the newest usable runs of one workflow inside the day window.

    Uses the workflow-runs endpoint rather than `gh run list`, which returns a
    different set of runs for different `--limit` values and does not always
    include the newest ones. The window and the page size are applied by the API,
    which documents this endpoint as newest first.

    Returns the runs and why the workflow could not be listed, which is empty on
    success. A workflow that cannot be listed has to be told apart from one that
    had nothing to report.
    """
    cutoff = f"{date.today() - timedelta(days=days):%Y-%m-%d}"
    completed = subprocess.run(
        [
            "gh",
            "api",
            "--method",
            "GET",
            f"repos/{repo}/actions/workflows/{workflow}/runs",
            "-f",
            f"per_page={FETCH_LIMIT}",
            "-f",
            "status=completed",
            "-f",
            f"created=>={cutoff}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return [], f"could not list runs: {first_line(completed)}"
    entries = sorted(
        json.loads(completed.stdout).get("workflow_runs", []),
        key=lambda entry: str(entry.get("created_at") or ""),
        reverse=True,
    )
    in_window = [
        entry for entry in entries if str(entry.get("created_at") or "") >= cutoff
    ]
    # The endpoint is asked for the window, but a page of older runs has been
    # seen in the wild. Dropping those runs would make a workflow look like it
    # had nothing to report, which is the kind of gap this report exists to
    # expose.
    if entries and not in_window:
        newest = str(entries[0].get("created_at") or "unknown date")[:10]
        return [], f"no run newer than {cutoff} came back, newest was {newest}"
    return [
        RunRef(
            workflow=workflow,
            run_id=str(entry["id"]),
            created_at=str(entry.get("created_at") or ""),
            conclusion=str(entry.get("conclusion") or ""),
            head_branch=str(entry.get("head_branch") or ""),
            retry_workflow=retry_only,
        )
        for entry in in_window
        if str(entry.get("conclusion") or "") in USABLE_CONCLUSIONS
    ][:limit], ""


def first_line(completed) -> str:
    """Return the most useful single line of a failed command's output."""
    output = completed.stderr.strip() or completed.stdout.strip()
    return output.splitlines()[0] if output else "no output"


def download_runs(runs: list[RunRef], dest: Path) -> None:
    """Download the per-case artifacts, one directory per run.

    A run with nothing to download is recorded on the run, so the summary can say
    why a run is missing instead of presenting it as a run without failures.
    """
    for run in runs:
        # Only the two artifact families that carry per-case results. Asking for
        # a whole run would pull every image and build artifact too.
        completed = subprocess.run(
            [
                "gh",
                "run",
                "download",
                run.run_id,
                "--pattern",
                "ci-result-*",
                "--pattern",
                "e2e-evidence-*",
                "--dir",
                str(dest / run.directory),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            # A cancelled or failed run can finish before any e2e job uploaded.
            run.note = f"no download: {first_line(completed)}"
            print(f"{run.workflow} {run.run_id}: {run.note}", file=sys.stderr)


def collect_runs(
    workflows: list[str], retry_workflows: list[str], args
) -> tuple[list[RunRef], list[str]]:
    """Return the runs to read and the workflows that could not be listed."""
    repo = repo_slug()
    runs: list[RunRef] = []
    problems: list[str] = []
    sources = [
        *((workflow, False) for workflow in workflows),
        *((workflow, True) for workflow in retry_workflows),
    ]
    for workflow, retry_only in sources:
        found, problem = recent_runs(
            workflow, args.runs, args.days, repo, retry_only=retry_only
        )
        runs += found
        if problem:
            problems.append(f"`{workflow}`: {problem}")
    download_runs(runs, args.download_dir)
    return runs, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        action="append",
        dest="workflows",
        help="Merged-code workflow to read. Repeatable, defaults to main.yml and nightly-build.yml.",
    )
    parser.add_argument(
        "--retry-workflow",
        action="append",
        dest="retry_workflows",
        help="Unmerged-code workflow whose runs count towards retries only. Repeatable, defaults to pr.yml.",
    )
    parser.add_argument(
        "--runs", type=int, default=40, help="Runs to keep per workflow at most."
    )
    parser.add_argument(
        "--days", type=int, default=14, help="Only read runs newer than this."
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument(
        "--from-dir", type=Path, help="Read an existing download instead of calling gh."
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be 1 or greater")
    if args.days < 1:
        parser.error("--days must be 1 or greater")

    workflows = args.workflows or list(DEFAULT_WORKFLOWS)
    retry_workflows = args.retry_workflows or list(DEFAULT_RETRY_WORKFLOWS)
    if args.from_dir is not None:
        root, runs, problems = args.from_dir, [], []
    else:
        root = args.download_dir
        runs, problems = collect_runs(workflows, retry_workflows, args)

    samples = load_samples(root, runs)
    text = render(aggregate(samples), samples, runs, problems)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
