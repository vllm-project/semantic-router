from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

import image_artifacts  # noqa: E402
from validate_workflows import Workflow, load_workflows  # noqa: E402
from workflow_policy_validation import validate_nightly_qualification  # noqa: E402

# Answers the gh calls the nightly workflows make and logs each one.
GH_STUB = """#!/usr/bin/env bash
echo "$*" >> "$GH_LOG"
case "$*" in
  *commits/main*) echo "$STUB_MAIN_SHA" ;;
  *nightly-build.yml/runs*) echo "$STUB_PASSED" ;;
  *actions/runs/*) echo "${STUB_ATTEMPT:-1}" ;;
  "run list"*) echo "$STUB_ACTIVE" ;;
esac
"""


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def events(data: dict) -> dict:
    # YAML 1.1 reads the `on` key as true.
    return data["on"] if "on" in data else data[True]


def run_step(
    script: str, env: dict[str, str], check: bool = True
) -> tuple[list[str], dict[str, str], int]:
    """Run a step's script against the gh stub; return gh calls, outputs and exit code."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stub = root / "gh"
        stub.write_text(GH_STUB)
        stub.chmod(0o755)
        log, output = root / "gh.log", root / "output"
        log.touch()
        output.touch()
        process = subprocess.run(
            ["bash", "-e", "-c", script],
            env={
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "GH_LOG": str(log),
                "GITHUB_OUTPUT": str(output),
                "GITHUB_REPOSITORY": "vllm-project/semantic-router",
                **env,
            },
            check=check,
            capture_output=True,
            text=True,
        )
        outputs = dict(
            line.split("=", 1)
            for line in output.read_text().splitlines()
            if "=" in line
        )
        return log.read_text().splitlines(), outputs, process.returncode


class NightlyQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nightly = load("nightly-build.yml")
        self.head = self.nightly["jobs"]["head"]["steps"][0]["run"]
        self.env = {"GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "7"}

    def test_repository_workflows_satisfy_the_nightly_policy(self) -> None:
        errors: list[str] = []
        validate_nightly_qualification(load_workflows(errors), errors)
        self.assertEqual(errors, [])

    def test_scheduled_run_skips_a_commit_another_run_covers(self) -> None:
        for covered, qualify in (("1", "false"), ("0", "true")):
            with self.subTest(covered=covered):
                calls, outputs, _ = run_step(
                    self.head,
                    {
                        **self.env,
                        "GITHUB_EVENT_NAME": "schedule",
                        "STUB_PASSED": covered,
                    },
                )
                self.assertEqual(outputs["qualify"], qualify)
                self.assertIn(f"head_sha={'a' * 40}", calls[0])
                # Its own run is still in progress and must not count.
                self.assertIn("select(.id != 7)", calls[0])
                self.assertIn(
                    '.conclusion == "success" or .status != "completed"', calls[0]
                )

    def test_dispatch_always_qualifies_under_one_date(self) -> None:
        calls, outputs, _ = run_step(
            self.head,
            {**self.env, "GITHUB_EVENT_NAME": "workflow_dispatch", "STUB_PASSED": "1"},
        )
        self.assertEqual((calls, outputs["qualify"]), ([], "true"))
        self.assertRegex(outputs["date"], r"^\d{8}$")
        for job in ("ci", "helm"):
            self.assertEqual(
                self.nightly["jobs"][job]["with"]["date"],
                "${{ needs.head.outputs.date }}",
            )

    def test_a_dispatch_cancels_a_running_qualification_and_a_schedule_waits(
        self,
    ) -> None:
        concurrency = self.nightly["jobs"]["ci"]["concurrency"]
        self.assertEqual(concurrency["group"], "nightly-qualification")
        expression = concurrency["cancel-in-progress"]
        expression = expression.removeprefix("${{").removesuffix("}}").strip()
        for event, cancels in (("workflow_dispatch", True), ("schedule", False)):
            with self.subTest(event=event):
                self.assertEqual(
                    eval(
                        expression.replace("github.event_name", repr(event)),
                        {"__builtins__": {}},
                    ),
                    cancels,
                )

    def test_nightly_chart_requires_the_runs_date(self) -> None:
        steps = load("helm-publish.yml")["jobs"]["package"]["steps"]
        script = next(step["run"] for step in steps if step.get("id") == "versions")
        _, outputs, _ = run_step(script, {"MODE": "nightly", "RUN_DATE": "20261105"})
        self.assertEqual(
            outputs,
            {
                "chart_version": "0.0.0-nightly.20261105",
                "app_version": "nightly-20261105",
            },
        )
        _, _, code = run_step(script, {"MODE": "nightly", "RUN_DATE": ""}, check=False)
        self.assertNotEqual(code, 0)

    def test_policy_rejects_weekly_runs_cancellable_publication_and_two_dates(
        self,
    ) -> None:
        errors: list[str] = []
        workflows = load_workflows(errors)
        data = copy.deepcopy(workflows["nightly-build.yml"].data)
        events(data)["schedule"] = [{"cron": "0 2 * * 1"}]
        data["concurrency"] = {"group": "nightly", "cancel-in-progress": True}
        del data["jobs"]["helm"]["with"]["date"]
        workflows["nightly-build.yml"] = Workflow(
            workflows["nightly-build.yml"].path, data
        )
        validate_nightly_qualification(workflows, errors)
        self.assertIn("nightly-build.yml must qualify main once every day", errors)
        self.assertIn(
            "nightly-build.yml must not let a newer run cancel an older run's publication",
            errors,
        )
        self.assertIn(
            "nightly-build.yml: 'helm' must use the run's one publication date", errors
        )


class NightlyRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retry = load("nightly-retry.yml")
        self.job = self.retry["jobs"]["retry"]
        self.script = self.job["steps"][0]["run"]
        self.env = {"RUN_ID": "41376111657", "RUN_SHA": "a" * 40, "STUB_ACTIVE": "0"}

    def test_retries_only_a_first_failed_main_attempt(self) -> None:
        trigger = events(self.retry)["workflow_run"]
        self.assertEqual(
            trigger,
            {"workflows": [load("nightly-build.yml")["name"]], "types": ["completed"]},
        )
        for condition in (
            "github.event.workflow_run.conclusion == 'failure'",
            "github.event.workflow_run.run_attempt == 1",
            "github.event.workflow_run.head_branch == 'main'",
        ):
            self.assertIn(condition, self.job["if"])
        self.assertEqual(
            self.job["permissions"], {"contents": "read", "actions": "write"}
        )

    def test_reruns_the_failed_jobs_of_mains_newest_qualification(self) -> None:
        calls, _, _ = run_step(self.script, {**self.env, "STUB_MAIN_SHA": "a" * 40})
        self.assertEqual(
            calls[-1],
            "run rerun 41376111657 --failed --repo vllm-project/semantic-router",
        )

    def test_stands_down_for_a_manual_rerun_or_a_newer_qualification(self) -> None:
        for name, env in (
            ("re-run by hand", {"STUB_MAIN_SHA": "a" * 40, "STUB_ATTEMPT": "2"}),
            ("main moved", {"STUB_MAIN_SHA": "b" * 40}),
            ("another run active", {"STUB_MAIN_SHA": "a" * 40, "STUB_ACTIVE": "1"}),
        ):
            with self.subTest(name):
                calls, _, _ = run_step(self.script, {**self.env, **env})
                self.assertFalse(
                    [call for call in calls if call.startswith("run rerun")]
                )


HELM_STUB = """#!/usr/bin/env bash
echo "$*" >> "$HELM_LOG"
if [[ "$1 $2" == "show chart" ]]; then
  if [[ "$STUB_CHART" == published ]]; then exit 0; fi
  echo "Error: $STUB_CHART" >&2
  exit 1
fi
"""


class PublishedNightlyDateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = load("nightly-build.yml")["jobs"]
        steps = load("helm-publish.yml")["jobs"]["publish"]["steps"]
        script = next(
            step["run"] for step in steps if step.get("name") == "Publish chart"
        )
        self.publish_chart = re.sub(r"\$\{\{[^}]*\}\}", "ghcr.io", script)

    def test_publishers_wait_for_the_date_check(self) -> None:
        reserve = self.jobs["reserve"]
        self.assertEqual(set(reserve["needs"]), {"head", "ci"})
        script = reserve["steps"][-1]["run"]
        self.assertIn("tools/ci/nightly_tags.py", script)
        self.assertEqual(
            reserve["steps"][-1]["env"]["DATE"], "${{ needs.head.outputs.date }}"
        )
        for job in ("docker", "helm"):
            self.assertIn("reserve", self.jobs[job]["needs"])
            self.assertIn(
                "needs.reserve.outputs.publish == 'true'", self.jobs[job]["if"]
            )

    def run_publish(self, mode: str, chart: str) -> tuple[list[str], int]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub, log = root / "helm", root / "helm.log"
            stub.write_text(HELM_STUB)
            stub.chmod(0o755)
            log.touch()
            process = subprocess.run(
                ["bash", "-e", "-c", self.publish_chart],
                env={
                    **os.environ,
                    "PATH": f"{root}:{os.environ['PATH']}",
                    "HELM_LOG": str(log),
                    "RUNNER_TEMP": str(root),
                    "MODE": mode,
                    "VERSION": "0.0.0-nightly.20261105",
                    "CHART": "semantic-router-0.0.0-nightly.20261105.tgz",
                    "STUB_CHART": chart,
                },
                capture_output=True,
                text=True,
                check=False,
            )
            return log.read_text().splitlines(), process.returncode

    def test_a_published_nightly_chart_is_never_pushed_again(self) -> None:
        for mode, chart, pushed, code in (
            ("nightly", "published", False, 0),
            ("nightly", "0.0.0-nightly.20261105: not found", True, 0),
            ("nightly", "unauthorized: authentication required", False, 1),
            ("main", "published", True, 0),
        ):
            with self.subTest(mode=mode, chart=chart):
                calls, returncode = self.run_publish(mode, chart)
                self.assertEqual(any(c.startswith("push ") for c in calls), pushed)
                self.assertEqual(returncode, code)


class RunDateTests(unittest.TestCase):
    def test_images_take_the_runs_date(self) -> None:
        with mock.patch.dict(os.environ, {"CI_RUN_DATE": "20261105"}):
            self.assertEqual(image_artifacts.run_date(), "20261105")
        with (
            mock.patch.dict(os.environ, {"CI_RUN_DATE": "2026-11-05"}),
            self.assertRaises(ValueError),
        ):
            image_artifacts.run_date()
        with mock.patch.dict(os.environ, {"CI_RUN_DATE": ""}):
            self.assertTrue(re.fullmatch(r"\d{8}", image_artifacts.run_date()))


if __name__ == "__main__":
    unittest.main()
