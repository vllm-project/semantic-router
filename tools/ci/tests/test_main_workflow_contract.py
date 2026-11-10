from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from workflow_policy_validation import validate_main_latest  # noqa: E402

UPSTREAM = "vllm-project/semantic-router"
CHART = "deploy/helm/semantic-router"


def evaluate(expression: str, values: dict[str, object]) -> object:
    """Evaluate a GitHub expression of comparisons joined by && and ||."""
    python = expression.removeprefix("${{").removesuffix("}}").strip()
    for key, value in values.items():
        python = python.replace(key, repr(value))
    python = python.replace("&&", " and ").replace("||", " or ")
    return eval(python, {"__builtins__": {}}, {})


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "ci",
            "GIT_AUTHOR_EMAIL": "ci@example.com",
            "GIT_COMMITTER_NAME": "ci",
            "GIT_COMMITTER_EMAIL": "ci@example.com",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class MainPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/main.yml").read_text(encoding="utf-8")
        )
        self.jobs = self.workflow["jobs"]

    def test_every_upstream_push_publishes_and_a_dispatch_qualifies_fully_first(
        self,
    ) -> None:
        call = self.jobs["ci"]["with"]
        for event, repository, ref, requested, publish, full in (
            ("push", UPSTREAM, "refs/heads/main", None, True, False),
            ("push", "fork/semantic-router", "refs/heads/main", None, False, False),
            ("workflow_dispatch", UPSTREAM, "refs/heads/main", True, True, True),
            ("workflow_dispatch", UPSTREAM, "refs/heads/main", False, False, False),
            ("workflow_dispatch", UPSTREAM, "refs/heads/feature", True, False, False),
        ):
            with self.subTest(event=event, repository=repository, ref=ref):
                values = {
                    "github.event_name": event,
                    "github.repository": repository,
                    "github.ref": ref,
                    "inputs.publish": requested,
                }
                self.assertEqual(bool(evaluate(call["publish"], values)), publish)
                self.assertEqual(bool(evaluate(call["full"], values)), full)

    def test_latest_moves_after_the_complete_image_set_one_run_at_a_time(
        self,
    ) -> None:
        self.assertNotIn("publish_latest", self.jobs["images"]["with"])
        latest = self.jobs["latest"]
        self.assertEqual(set(latest["needs"]), {"ci", "images"})
        self.assertIn("needs.images.result == 'success'", latest["if"])
        self.assertEqual(
            latest["concurrency"], {"group": "main-latest", "cancel-in-progress": False}
        )
        move = latest["steps"][-1]
        self.assertIn("tools/ci/latest_tags.py", move["run"])
        self.assertIn('--source "$GITHUB_SHA"', move["run"])
        self.assertEqual(
            move["env"]["IMAGES"], "${{ needs.ci.outputs.publish_images }}"
        )
        # The development CLI pins its own commit's images, never latest.
        self.assertEqual(set(self.jobs["pypi"]["needs"]), {"ci", "images"})
        workflow = SimpleNamespace(jobs=self.jobs)
        errors: list[str] = []
        validate_main_latest({"main.yml": workflow}, errors)
        self.assertEqual(errors, [])
        for name, change in (
            ("latest with the images", ("images", "with", "publish_latest", True)),
            ("before the images", ("latest", "needs", None, ["ci"])),
            ("a cancelled move", ("latest", "concurrency", "cancel-in-progress", True)),
            ("runs in parallel", ("latest", "concurrency", "group", "main-{0}")),
        ):
            with self.subTest(name):
                jobs = copy.deepcopy(self.jobs)
                job, field, key, value = change
                if key is None:
                    jobs[job][field] = value
                else:
                    jobs[job][field][key] = value
                errors = []
                validate_main_latest({"main.yml": SimpleNamespace(jobs=jobs)}, errors)
                self.assertEqual(len(errors), 1)


class DevelopmentChartTests(unittest.TestCase):
    def setUp(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/helm-publish.yml").read_text(
                encoding="utf-8"
            )
        )
        self.jobs = workflow["jobs"]
        self.step = next(
            step
            for step in self.jobs["package"]["steps"]
            if step.get("id") == "current"
        )

    def test_publishes_only_the_chart_mains_head_carries(self) -> None:
        self.assertEqual(self.step["if"], "inputs.mode == 'main'")
        self.assertIn(
            "needs.package.outputs.current != 'false'", self.jobs["publish"]["if"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, work = root / "upstream", root / "work"
            upstream.mkdir()
            git(upstream, "init", "-q", "-b", "main")
            (upstream / CHART).mkdir(parents=True)
            (upstream / CHART / "values.yaml").write_text("replicas: 1\n")
            git(upstream, "add", "-A")
            git(upstream, "commit", "-q", "-m", "chart")
            git(root, "clone", "-q", "--depth=1", f"file://{upstream}", str(work))
            for change, current in (
                ("", "true"),
                ("README.md", "true"),
                (f"{CHART}/values.yaml", "false"),
            ):
                with self.subTest(newer_main_change=change):
                    if change:
                        (upstream / change).write_text(f"{change} changed\n")
                        git(upstream, "add", "-A")
                        git(upstream, "commit", "-q", "-m", change)
                    output = root / "output"
                    output.write_text("")
                    subprocess.run(
                        ["bash", "-e", "-c", self.step["run"]],
                        cwd=work,
                        env={
                            **os.environ,
                            "CHART_PATH": CHART,
                            "GITHUB_OUTPUT": str(output),
                            "GIT_CONFIG_GLOBAL": os.devnull,
                            "GIT_CONFIG_NOSYSTEM": "1",
                        },
                        check=True,
                        capture_output=True,
                    )
                    self.assertEqual(output.read_text(), f"current={current}\n")


class MainWorkflowContractTests(unittest.TestCase):
    def test_main_validation_runs_cannot_be_coalesced(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "main.yml").read_text(
                encoding="utf-8"
            )
        )

        self.assertNotIn(
            "concurrency",
            workflow,
            "main push runs must remain per-commit so change classification cannot skip a commit",
        )
        group = workflow["jobs"]["ci"]["concurrency"]["group"]
        expression = group.removeprefix("${{").removesuffix("}}").strip()
        for event, publish, expected in (
            ("push", None, "main-7"),
            ("workflow_dispatch", False, "main-7"),
            ("workflow_dispatch", True, "main-publication"),
        ):
            with self.subTest(event=event, publish=publish):
                python = (
                    expression.replace("github.event_name", repr(event))
                    .replace("inputs.publish", repr(publish))
                    .replace("github.run_id", "'7'")
                    .replace("&&", " and ")
                    .replace("||", " or ")
                )
                actual = eval(
                    python,
                    {"__builtins__": {}},
                    {"format": lambda text, value: text.replace("{0}", value)},
                )
                self.assertEqual(actual, expected)
        self.assertTrue(workflow["jobs"]["ci"]["concurrency"]["cancel-in-progress"])

    def test_pypi_accepts_only_qualified_main_publication_or_stable_tag(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/pypi-publish.yml").read_text()
        )
        condition = workflow["jobs"]["pypi"]["if"]
        # Evaluate the workflow's actual guard across publication boundaries.
        for event, ref, channel, prebuilt, allowed in (
            ("workflow_dispatch", "refs/heads/main", "dev", True, True),
            ("workflow_dispatch", "refs/heads/main", "dev", False, False),
            ("workflow_dispatch", "refs/heads/feature", "dev", True, False),
            ("push", "refs/heads/main", "dev", True, True),
            ("push", "refs/heads/main", "dev", False, False),
            ("push", "refs/heads/feature", "dev", True, False),
            ("pull_request", "refs/heads/main", "dev", True, False),
            ("push", "refs/tags/v0.5.0", "stable", True, True),
            ("workflow_dispatch", "refs/tags/v0.5.0", "stable", True, False),
        ):
            with self.subTest(event=event, ref=ref, channel=channel, prebuilt=prebuilt):
                expression = condition
                for key, value in {
                    "inputs.build-only": False,
                    "inputs.prebuilt-dist": prebuilt,
                    "inputs.channel": channel,
                    "needs.build.result": "success",
                    "github.repository": "vllm-project/semantic-router",
                    "github.event_name": event,
                    "github.ref": ref,
                }.items():
                    expression = expression.replace(key, repr(value))
                expression = expression.replace("&&", " and ").replace("||", " or ")
                expression = expression.replace("!", "not ")
                self.assertEqual(
                    eval(
                        expression,
                        {"__builtins__": {}},
                        {
                            "always": lambda: True,
                            "cancelled": lambda: False,
                            "failure": lambda: False,
                            "startsWith": lambda value, prefix: value.startswith(
                                prefix
                            ),
                        },
                    ),
                    allowed,
                )


class PullRequestGateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "pr.yml").read_text(encoding="utf-8")
        )
        self.gate = self.workflow["jobs"]["pr-gate"]

    def test_gates_reconcile_failures_but_stop_when_the_workflow_is_cancelled(
        self,
    ) -> None:
        for filename, job in (
            ("pr.yml", "pr-gate"),
            ("ci.yml", "gate"),
        ):
            with self.subTest(workflow=filename):
                workflow = yaml.safe_load(
                    (REPO_ROOT / ".github" / "workflows" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                expression = (
                    workflow["jobs"][job]["if"]
                    .removeprefix("${{")
                    .removesuffix("}}")
                    .strip()
                    .replace("!", "not ")
                )
                for cancelled in (False, True):
                    for result in ("success", "failure", "skipped"):
                        with self.subTest(cancelled=cancelled, prerequisite=result):
                            allowed = eval(
                                expression,
                                {"__builtins__": {}},
                                {
                                    "always": lambda: True,
                                    "cancelled": lambda cancelled=cancelled: cancelled,
                                    "success": lambda result=result: result
                                    == "success",
                                    "failure": lambda result=result: result
                                    == "failure",
                                },
                            )
                            self.assertEqual(allowed, not cancelled)
        self.assertTrue(self.workflow["concurrency"]["cancel-in-progress"])

    def test_stable_gate_rejects_unsuccessful_shared_qualification(self) -> None:
        script = self.gate["steps"][0]["run"]
        self.assertEqual(self.gate["needs"], "ci")
        for result, expected_exit in (
            ("success", 0),
            ("skipped", 1),
            ("failure", 1),
            ("cancelled", 1),
        ):
            with self.subTest(result=result):
                actual = subprocess.run(
                    ["bash", "-e", "-c", script],
                    env={**os.environ, "RESULT": result},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(actual.returncode, expected_exit, actual.stderr)


if __name__ == "__main__":
    unittest.main()
