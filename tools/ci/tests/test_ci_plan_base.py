from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"
sys.path.insert(0, str(ROOT / "tools/ci"))
from ci_plan import github_outputs, make_plan  # noqa: E402
from classify_pr_changes import classify, git_changed_files  # noqa: E402

MAIN_FILE = "config/recipes/maintained/example/recipe.yaml"
PR_FILE = "website/docs/overview.md"
ZERO_SHA = "0" * 40


def load_workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


class MergeRefHistory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q", "-b", "main")
        self.write("README.md", "base\n")
        self.base = self.commit("base")
        self.git("checkout", "-q", "-b", "pr")
        self.write(PR_FILE, "pull request\n")
        self.pr = self.commit("pull request")
        self.git("checkout", "-q", "main")
        self.write(MAIN_FILE, "newer main\n")
        self.main = self.commit("newer main")
        self.git("merge", "-q", "--no-ff", "-m", "merge ref", "pr")
        self.merge = self.git("rev-parse", "HEAD")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
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

    def write(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def changed_files(self, base: str) -> list[str]:
        with contextlib.chdir(self.root):
            return git_changed_files(base, self.merge)


class PlanBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.history = MergeRefHistory(Path(directory.name))
        steps = load_workflow("ci-changes.yml")["jobs"]["plan"]["steps"]
        self.steps = {step["id"]: step for step in steps if "id" in step}

    def resolve_base(self, event: str, before: str) -> str:
        output = self.history.root / "github-output"
        output.write_text("", encoding="utf-8")
        subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", self.steps["base"]["run"]],
            cwd=self.history.root,
            env={
                **os.environ,
                "GITHUB_EVENT_NAME": event,
                "GITHUB_SHA": self.history.merge,
                "GITHUB_OUTPUT": str(output),
                "EVENT_BEFORE": before,
            },
            check=True,
            capture_output=True,
            text=True,
        )
        return output.read_text(encoding="utf-8").removeprefix("sha=").strip()

    def test_stale_event_base_selects_main_changes(self) -> None:
        paths = self.history.changed_files(self.history.base)
        self.assertEqual(sorted(paths), sorted([MAIN_FILE, PR_FILE]))
        self.assertIn("recipe-conformance", classify(paths).selected_jobs)

    def test_pull_request_base_is_the_merge_ref_first_parent(self) -> None:
        base = self.resolve_base("pull_request", "")
        self.assertEqual(base, self.history.main)
        paths = self.history.changed_files(base)
        self.assertEqual(paths, [PR_FILE])
        self.assertNotIn("recipe-conformance", classify(paths).selected_jobs)

    def test_other_events_keep_the_pushed_before_sha(self) -> None:
        for event, before in (
            ("push", self.history.base),
            ("push", ZERO_SHA),
            ("schedule", ""),
            ("workflow_dispatch", ""),
        ):
            with self.subTest(event=event, before=before):
                self.assertEqual(self.resolve_base(event, before), before)

    def test_plan_step_uses_only_the_resolved_base(self) -> None:
        self.assertEqual(
            self.steps["base"]["env"], {"EVENT_BEFORE": "${{ github.event.before }}"}
        )
        self.assertEqual(
            self.steps["plan"]["env"]["BASE_SHA"], "${{ steps.base.outputs.sha }}"
        )
        self.assertIn('--base "$BASE_SHA"', self.steps["plan"]["run"])


class PerformanceBaseTests(unittest.TestCase):
    def test_plan_output_carries_the_compared_base(self) -> None:
        plan = make_plan([PR_FILE], source_sha="a" * 40, base_sha="b" * 40)
        self.assertEqual(json.loads(github_outputs(plan)["plan"])["base_sha"], "b" * 40)

    def test_performance_baseline_reuses_the_plan_base(self) -> None:
        job = load_workflow("ci.yml")["jobs"]["performance"]
        self.assertIn("plan", job["needs"])
        self.assertEqual(
            job["with"]["base-ref"],
            "${{ inputs.performance-base-ref"
            " || fromJSON(needs.plan.outputs.plan).base_sha || 'HEAD^' }}",
        )
        steps = load_workflow("performance-test.yml")["jobs"]["component-benchmarks"][
            "steps"
        ]
        baseline = next(
            step for step in steps if "PERF_BASE_REF" in step.get("env", {})
        )
        self.assertEqual(
            baseline["env"]["PERF_BASE_REF"], "${{ inputs.base-ref || 'HEAD^' }}"
        )

    def test_planned_workflows_never_read_the_event_base(self) -> None:
        for name in ("ci-changes.yml", "ci.yml", "performance-test.yml"):
            with self.subTest(workflow=name):
                self.assertNotIn(
                    "pull_request.base.sha",
                    (WORKFLOWS / name).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
