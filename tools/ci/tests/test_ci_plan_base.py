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


class GitHistory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q", "-b", "main")

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


class MergeRefHistory(GitHistory):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
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
            self.steps["base"]["env"],
            {
                "EVENT_BEFORE": "${{ github.event.before }}",
                "PROFILE": "${{ inputs.profile }}",
                "GH_TOKEN": "${{ github.token }}",
            },
        )
        self.assertEqual(
            self.steps["plan"]["env"]["BASE_SHA"], "${{ steps.base.outputs.sha }}"
        )
        self.assertIn('--base "$BASE_SHA"', self.steps["plan"]["run"])

    def publisher_ready(self, event: str = "pull_request", source: str = "") -> bool:
        output = self.history.root / "publisher-output"
        output.write_text("", encoding="utf-8")
        subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", self.steps["publisher"]["run"]],
            cwd=self.history.root,
            env={
                **os.environ,
                "GITHUB_EVENT_NAME": event,
                "GITHUB_SHA": source or self.history.merge,
                "GITHUB_OUTPUT": str(output),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        return output.read_text(encoding="utf-8").strip() == "ready=true"

    def write_publisher(self, *, script: bool = True) -> None:
        self.history.write(".github/workflows/ci-status.yml", "name: trusted\n")
        if script:
            self.history.write("tools/ci/pr_ci_status.py", "# trusted publisher\n")

    def merge_publisher_pr(self, *, publisher_on_base: bool) -> str:
        if publisher_on_base:
            self.write_publisher()
            self.history.commit("trusted publisher on main")
        self.history.git("checkout", "-q", "-b", "status-pr")
        self.history.write(PR_FILE, "updated pull request\n")
        if not publisher_on_base:
            self.write_publisher()
        self.history.commit("pull request status changes")
        self.history.git("checkout", "-q", "main")
        self.history.git("merge", "-q", "--no-ff", "-m", "merge ref", "status-pr")
        return self.history.git("rev-parse", "HEAD")

    def test_pr_cannot_enable_deferred_success_by_adding_its_own_publisher(
        self,
    ) -> None:
        source = self.merge_publisher_pr(publisher_on_base=False)
        self.assertFalse(self.publisher_ready(source=source))

    def test_publisher_in_actual_merge_base_enables_waiting(self) -> None:
        source = self.merge_publisher_pr(publisher_on_base=True)
        self.assertTrue(self.publisher_ready(source=source))

    def test_old_merge_base_stays_strict_after_main_gains_publisher(self) -> None:
        self.write_publisher()
        self.history.commit("publisher added after this PR merge ref")
        self.assertFalse(self.publisher_ready())

    def test_workflow_without_its_script_does_not_enable_waiting(self) -> None:
        self.write_publisher(script=False)
        self.history.commit("incomplete publisher")
        self.history.git("checkout", "-q", "-b", "incomplete-pr")
        self.history.write(PR_FILE, "another change\n")
        self.history.commit("pull request")
        self.history.git("checkout", "-q", "main")
        self.history.git("merge", "-q", "--no-ff", "-m", "merge ref", "incomplete-pr")
        self.assertFalse(
            self.publisher_ready(source=self.history.git("rev-parse", "HEAD"))
        )

    def test_non_pr_events_never_enable_deferred_success(self) -> None:
        source = self.merge_publisher_pr(publisher_on_base=True)
        for event in ("push", "schedule", "workflow_dispatch", "merge_group"):
            with self.subTest(event=event):
                self.assertFalse(self.publisher_ready(event=event, source=source))


# Answers main_push_base.py's two GitHub API reads from STUB_RUNS and STUB_GATES.
GH_STUB = """#!/usr/bin/env python3
import json, os, re, sys
if os.environ.get("STUB_FAIL"):
    sys.exit("HTTP 502")
path = sys.argv[2]
if "/workflows/main.yml/runs" in path:
    print(json.dumps({"workflow_runs": json.loads(os.environ["STUB_RUNS"])}))
else:
    run = re.search(r"/runs/(\\d+)/jobs", path).group(1)
    gate = json.loads(os.environ["STUB_GATES"]).get(run)
    print(json.dumps({"jobs": [{"name": "CI / Gate", "conclusion": gate}]}))
"""


class SupersededPushBaseTests(unittest.TestCase):
    """A newer push cancels older qualification, so its plan reaches further back."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.history = GitHistory(self.root)
        self.commits = []
        for index in range(5):
            self.history.write(f"file-{index}.txt", f"{index}\n")
            self.commits.append(self.history.commit(f"main {index}"))
        script = self.root / "tools/ci/main_push_base.py"
        script.parent.mkdir(parents=True)
        script.write_text((ROOT / "tools/ci/main_push_base.py").read_text())
        stub = self.root / "bin/gh"
        stub.parent.mkdir()
        stub.write_text(GH_STUB)
        stub.chmod(0o755)
        steps = load_workflow("ci-changes.yml")["jobs"]["plan"]["steps"]
        self.script = next(step for step in steps if step.get("id") == "base")["run"]

    def resolve(self, gates: dict[str, str | None], **env: str) -> tuple[str, str]:
        """Run the base step for a push of the newest commit; runs are numbered by commit."""
        runs = [
            {"id": index, "head_sha": sha} for index, sha in enumerate(self.commits)
        ]
        output = self.root / "github-output"
        output.write_text("", encoding="utf-8")
        process = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", self.script],
            cwd=self.root,
            env={
                **os.environ,
                "PATH": f"{self.root / 'bin'}:{os.environ['PATH']}",
                "PROFILE": "main",
                "GITHUB_EVENT_NAME": "push",
                "GITHUB_REPOSITORY": "o/r",
                "GITHUB_SHA": self.commits[4],
                "GITHUB_OUTPUT": str(output),
                "EVENT_BEFORE": self.commits[3],
                "STUB_RUNS": json.dumps(runs),
                "STUB_GATES": json.dumps(gates),
                **env,
            },
            check=True,
            capture_output=True,
            text=True,
        )
        return output.read_text().removeprefix("sha=").strip(), process.stderr

    def test_compares_with_before_when_nothing_was_cancelled(self) -> None:
        for conclusion in ("success", "failure"):
            with self.subTest(previous_gate=conclusion):
                base, _ = self.resolve({"3": conclusion})
                self.assertEqual(base, self.commits[3])

    def test_covers_every_commit_a_newer_push_cancelled(self) -> None:
        base, _ = self.resolve({"3": "cancelled", "2": None, "1": "success"})
        self.assertEqual(base, self.commits[1])
        self.assertEqual(
            sorted(
                self.history.git("diff", "--name-only", base, self.commits[4]).split()
            ),
            ["file-2.txt", "file-3.txt", "file-4.txt"],
        )

    def test_falls_back_to_before_with_a_warning(self) -> None:
        for name, gates, env in (
            ("no finished checks", {"3": "cancelled"}, {}),
            ("API failure", {}, {"STUB_FAIL": "1"}),
        ):
            with self.subTest(name):
                base, stderr = self.resolve(gates, **env)
                self.assertEqual(base, self.commits[3])
                self.assertIn("::warning::", stderr)

    def test_other_profiles_keep_the_pushed_before_sha(self) -> None:
        base, _ = self.resolve({"1": "success"}, PROFILE="nightly")
        self.assertEqual(base, self.commits[3])


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
