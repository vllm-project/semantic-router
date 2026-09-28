from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

from validate_workflows import load_workflows, needs  # noqa: E402


class NetlifyPreviewWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        self.workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.workflow = self.workflows["netlify-preview.yml"]
        self.jobs = self.workflow.jobs

    def test_only_new_exact_pr_commands_can_request_a_preview(self) -> None:
        self.assertEqual(
            self.workflow.events, {"issue_comment": {"types": ["created"]}}
        )
        self.assertEqual(
            " ".join(self.jobs["authorize"]["if"].split()),
            "github.event.issue.pull_request != null "
            "&& github.event.comment.body == '/netlify'",
        )
        self.assertEqual(needs(self.jobs["build"]), {"authorize"})
        self.assertEqual(
            self.jobs["build"]["if"], "needs.authorize.outputs.build == 'true'"
        )
        command_step = next(
            step
            for step in self.jobs["authorize"]["steps"]
            if step.get("id") == "request"
        )
        self.assertEqual(
            command_step["run"],
            'python3 tools/ci/netlify_preview.py authorize --event "$GITHUB_EVENT_PATH"',
        )
        self.assertEqual(
            self.jobs["authorize"]["outputs"]["build"],
            "${{ steps.request.outputs.build }}",
        )

    def test_requests_serialize_per_pr_without_cancelling_active_builds(self) -> None:
        concurrency = self.workflow.data["concurrency"]
        self.assertFalse(concurrency["cancel-in-progress"])
        self.assertEqual(
            concurrency["group"],
            "netlify-preview-${{ github.event.issue.pull_request "
            "&& github.event.comment.body == '/netlify' "
            "&& github.event.issue.number || github.run_id }}",
            "unrelated comments must not replace a pending preview request",
        )

    def test_only_trusted_jobs_can_write_preview_statuses(self) -> None:
        self.assertEqual(self.workflow.data["permissions"], {"contents": "read"})
        self.assertEqual(
            self.jobs["authorize"]["permissions"],
            {
                "contents": "read",
                "pull-requests": "read",
                "actions": "read",
                "statuses": "write",
            },
        )
        self.assertEqual(self.jobs["build"]["permissions"], {"contents": "read"})
        self.assertEqual(
            self.jobs["publish"]["permissions"],
            {"contents": "read", "pull-requests": "read", "statuses": "write"},
        )

    def test_pr_code_builds_without_secrets_at_the_authorized_sha(self) -> None:
        self.assertNotIn("secrets.", str(self.workflow.data.get("env", {})))
        build = self.jobs["build"]
        self.assertNotIn("secrets.", str(build))
        self.assertNotIn("environment", build)
        checkout = next(
            step
            for step in build["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        self.assertEqual(checkout["with"]["ref"], "${{ needs.authorize.outputs.sha }}")
        self.assertEqual(
            checkout["with"]["repository"],
            "${{ needs.authorize.outputs.repository }}",
        )
        self.assertFalse(checkout["with"]["persist-credentials"])
        self.assertEqual(
            [step["run"] for step in build["steps"] if "run" in step],
            ["make docs-build"],
        )

    def test_publisher_runs_trusted_code_with_only_the_static_artifact(self) -> None:
        self.assertEqual(needs(self.jobs["publish"]), {"authorize", "build"})
        for job_name in ("authorize", "publish"):
            with self.subTest(job=job_name):
                checkouts = [
                    step
                    for step in self.jobs[job_name]["steps"]
                    if step.get("uses", "").startswith("actions/checkout@")
                ]
                self.assertEqual(len(checkouts), 1)
                self.assertEqual(checkouts[0]["with"]["ref"], "${{ github.sha }}")
                self.assertFalse(checkouts[0]["with"]["persist-credentials"])
                self.assertNotIn("repository", checkouts[0]["with"])
        upload = next(
            step
            for step in self.jobs["build"]["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        )
        download = next(
            step
            for step in self.jobs["publish"]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        )
        self.assertEqual(upload["with"]["path"], "website/build/")
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(download["with"]["name"], upload["with"]["name"])
        self.assertIn("github.run_id", upload["with"]["name"])
        self.assertIn("github.run_attempt", upload["with"]["name"])
        self.assertEqual(download["with"]["path"], "${{ runner.temp }}/netlify-site")
        self.assertNotIn("run-id", download["with"])
        self.assertEqual(
            [step["run"] for step in self.jobs["publish"]["steps"] if "run" in step],
            [
                "python3 tools/ci/netlify_preview.py publish "
                '--directory "$RUNNER_TEMP/netlify-site"'
            ],
            "the privileged runner must not evaluate PR build commands or config",
        )

    def test_failed_builds_and_downloads_still_reach_status_reporter(self) -> None:
        publish = self.jobs["publish"]
        self.assertEqual(
            publish["if"], "always() && needs.authorize.outputs.build == 'true'"
        )
        report = next(step for step in publish["steps"] if "run" in step)
        self.assertEqual(report["if"], "always()")
        self.assertEqual(
            report["env"]["BUILD_RESULT"],
            "${{ needs.build.result == 'success' "
            "&& steps.artifact.outcome == 'success' && 'success' || 'failure' }}",
        )

    def test_community_command_handler_leaves_netlify_to_its_owner(self) -> None:
        condition = self.workflows["community.yml"].jobs["issue-manager"]["if"]
        self.assertIn("github.event.comment.body != '/netlify'", condition)


class NetlifyAutomaticBuildTests(unittest.TestCase):
    def test_ignore_command_allows_only_production_main(self) -> None:
        config = tomllib.loads((REPO_ROOT / "netlify.toml").read_text("utf-8"))
        command = config["build"]["ignore"]
        for context, branch, expected_exit in (
            ("production", "main", 1),
            ("production", "feature", 0),
            ("deploy-preview", "main", 0),
            ("deploy-preview", "feature", 0),
            ("branch-deploy", "feature", 0),
            ("branch-deploy", "main", 0),
            (None, "main", 0),
            ("production", None, 0),
            (None, None, 0),
        ):
            with self.subTest(context=context, branch=branch):
                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if key not in {"CONTEXT", "BRANCH"}
                }
                if context is not None:
                    environment["CONTEXT"] = context
                if branch is not None:
                    environment["BRANCH"] = branch
                result = subprocess.run(
                    ["bash", "-c", command],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected_exit, result.stderr)


if __name__ == "__main__":
    unittest.main()
