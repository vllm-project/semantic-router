from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import Workflow, load_workflows  # noqa: E402
from workflow_policy_validation import validate_nightly_qualification  # noqa: E402

# Answers the gh calls the nightly workflows make and logs each one.
GH_STUB = """#!/usr/bin/env bash
echo "$*" >> "$GH_LOG"
case "$*" in
  *commits/main*) echo "$STUB_MAIN_SHA" ;;
  *nightly-build.yml/runs*) echo "$STUB_PASSED" ;;
  "run list"*) echo "$STUB_ACTIVE" ;;
esac
"""


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def events(data: dict) -> dict:
    # YAML 1.1 reads the `on` key as true.
    return data["on"] if "on" in data else data[True]


def run_step(script: str, env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Run a workflow step's script against the gh stub; return gh calls and outputs."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stub = root / "gh"
        stub.write_text(GH_STUB)
        stub.chmod(0o755)
        log, output = root / "gh.log", root / "output"
        log.touch()
        output.touch()
        subprocess.run(
            ["bash", "-e", "-c", script],
            env={
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "GH_LOG": str(log),
                "GITHUB_OUTPUT": str(output),
                "GITHUB_REPOSITORY": "vllm-project/semantic-router",
                **env,
            },
            check=True,
            capture_output=True,
            text=True,
        )
        outputs = dict(
            line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
        )
        return log.read_text().splitlines(), outputs


class NightlyQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nightly = load("nightly-build.yml")
        self.head = self.nightly["jobs"]["head"]["steps"][0]["run"]

    def test_repository_workflows_satisfy_the_nightly_policy(self) -> None:
        errors: list[str] = []
        validate_nightly_qualification(load_workflows(errors), errors)
        self.assertEqual(errors, [])

    def test_scheduled_run_skips_a_commit_another_run_covers(self) -> None:
        for covered, qualify in (("1", "false"), ("0", "true")):
            with self.subTest(covered=covered):
                calls, outputs = run_step(
                    self.head,
                    {
                        "GITHUB_EVENT_NAME": "schedule",
                        "GITHUB_SHA": "a" * 40,
                        "GITHUB_RUN_ID": "7",
                        "STUB_PASSED": covered,
                    },
                )
                self.assertEqual(outputs, {"qualify": qualify})
                self.assertIn(f"head_sha={'a' * 40}", calls[0])
                # Its own run is still in progress and must not count.
                self.assertIn("select(.id != 7)", calls[0])
                self.assertIn('.conclusion == "success" or .status != "completed"', calls[0])

    def test_dispatch_always_qualifies(self) -> None:
        calls, outputs = run_step(
            self.head,
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_SHA": "a" * 40,
                "GITHUB_RUN_ID": "7",
                "STUB_PASSED": "1",
            },
        )
        self.assertEqual((calls, outputs), ([], {"qualify": "true"}))

    def test_policy_rejects_weekly_runs_and_cancellable_publication(self) -> None:
        errors: list[str] = []
        workflows = load_workflows(errors)
        data = copy.deepcopy(workflows["nightly-build.yml"].data)
        events(data)["schedule"] = [{"cron": "0 2 * * 1"}]
        data["concurrency"] = {"group": "nightly", "cancel-in-progress": True}
        workflows["nightly-build.yml"] = Workflow(workflows["nightly-build.yml"].path, data)
        validate_nightly_qualification(workflows, errors)
        self.assertIn("nightly-build.yml must qualify main once every day", errors)
        self.assertIn(
            "nightly-build.yml must not let a newer run cancel an older run's publication",
            errors,
        )


class NightlyRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retry = load("nightly-retry.yml")
        self.job = self.retry["jobs"]["retry"]
        self.script = self.job["steps"][0]["run"]
        self.env = {"RUN_ID": "41376111657", "RUN_SHA": "a" * 40, "STUB_ACTIVE": "0"}

    def test_retries_only_a_first_failed_main_attempt(self) -> None:
        trigger = events(self.retry)["workflow_run"]
        self.assertEqual(trigger, {"workflows": [load("nightly-build.yml")["name"]], "types": ["completed"]})
        for condition in (
            "github.event.workflow_run.conclusion == 'failure'",
            "github.event.workflow_run.run_attempt == 1",
            "github.event.workflow_run.head_branch == 'main'",
        ):
            self.assertIn(condition, self.job["if"])
        self.assertEqual(self.job["permissions"], {"contents": "read", "actions": "write"})

    def test_reruns_the_failed_jobs_of_mains_newest_qualification(self) -> None:
        calls, _ = run_step(self.script, {**self.env, "STUB_MAIN_SHA": "a" * 40})
        self.assertEqual(
            calls[-1], "run rerun 41376111657 --failed --repo vllm-project/semantic-router"
        )

    def test_leaves_a_superseded_run_to_the_next_qualification(self) -> None:
        for name, env in (
            ("main moved", {"STUB_MAIN_SHA": "b" * 40}),
            ("another run active", {"STUB_MAIN_SHA": "a" * 40, "STUB_ACTIVE": "1"}),
        ):
            with self.subTest(name):
                calls, _ = run_step(self.script, {**self.env, **env})
                self.assertFalse([call for call in calls if call.startswith("run rerun")])


if __name__ == "__main__":
    unittest.main()
