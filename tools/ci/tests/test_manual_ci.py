from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

import manual_ci  # noqa: E402

REPO = "vllm-project/semantic-router"
HEAD, BASE, MERGE, TRUSTED = (character * 40 for character in "abcd")
CREATED = "2026-10-18T10:00:00Z"
REQUESTED = "2026-10-18T10:01:00Z"


def archive(value: dict, filename="authorization.json") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as output:
        output.writestr(filename, json.dumps(value))
    return stream.getvalue()


class FakeGitHub:
    def __init__(self):
        self.permission = "write"
        self.calls = []
        self.posts = []
        self.pull = {
            "state": "open",
            "head": {
                "sha": HEAD,
                "ref": "feature",
                "repo": {"full_name": "contributor/fork"},
            },
            "base": {"sha": BASE, "ref": "main", "repo": {"full_name": REPO}},
            "merge_commit_sha": MERGE,
        }
        self.base_ref = {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": BASE},
        }
        self.merge_parents = [{"sha": BASE}, {"sha": HEAD}]
        self.comment = {
            "id": 99,
            "body": "/ci run",
            "created_at": REQUESTED,
            "updated_at": REQUESTED,
            "issue_url": f"https://api.github.com/repos/{REPO}/issues/42",
            "user": {"login": "maintainer"},
        }
        self.pr_run = {
            "id": 123,
            "event": "pull_request",
            "head_sha": HEAD,
            "head_branch": "feature",
            "head_repository": {"full_name": "contributor/fork"},
            "created_at": CREATED,
            "pull_requests": [{"number": 42}],
            "status": "completed",
        }
        self.pr_runs = [self.pr_run]
        self.plan = {"source_sha": MERGE, "base_sha": BASE}
        self.plan["plan_sha256"] = hashlib.sha256(
            json.dumps(self.plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.controller = {
            "id": 456,
            "workflow_id": 10,
            "path": manual_ci.CONTROLLER,
            "event": "issue_comment",
            "head_branch": "main",
            "head_sha": TRUSTED,
            "head_repository": {"full_name": REPO},
            "actor": {"login": "maintainer"},
        }
        self.authorize_job = {
            "name": manual_ci.AUTHORIZE_JOB,
            "status": "completed",
            "conclusion": "success",
        }
        self.trusted_ancestor = TRUSTED
        self.artifacts = []
        self.comments = []
        self.additional_comments = {}
        self.zip_values = {}

    def request(self, endpoint, *, method="GET"):
        self.calls.append(endpoint)
        if method == "POST":
            self.posts.append(endpoint)
            return None
        if "/collaborators/" in endpoint:
            return {"permission": self.permission}
        if endpoint.endswith("/pulls/42"):
            return self.pull
        if endpoint.endswith("/git/ref/heads/main"):
            return self.base_ref
        if "/git/commits/" in endpoint:
            return {"parents": self.merge_parents}
        if endpoint.endswith("/issues/comments/99"):
            return self.comment
        if "/issues/comments/" in endpoint:
            return self.additional_comments[int(endpoint.rsplit("/", 1)[-1])]
        if "/issues/42/comments?" in endpoint:
            return self.comments
        if "/actions/workflows/pr.yml/runs?" in endpoint:
            return {"workflow_runs": self.pr_runs}
        if endpoint.endswith("/actions/runs/123"):
            return self.pr_run
        if endpoint.endswith(
            ("/actions/runs/456", "/actions/runs/457", "/actions/runs/458")
        ):
            return self.controller
        if endpoint.endswith("/actions/workflows/manual-ci.yml"):
            return {"id": 10}
        if "/compare/" in endpoint:
            return {"merge_base_commit": {"sha": self.trusted_ancestor}}
        if "/attempts/1/jobs?" in endpoint:
            return {"jobs": [self.authorize_job]}
        if "/actions/runs/123/artifacts?" in endpoint:
            return {"artifacts": [{"id": 1, "name": "ci-plan", "size_in_bytes": 1000}]}
        if "/actions/runs/124/artifacts?" in endpoint:
            return {"artifacts": []}
        if "/actions/artifacts?" in endpoint:
            return {"artifacts": self.artifacts}
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    def raw(self, endpoint):
        if endpoint.endswith("/artifacts/1/zip"):
            return archive(self.plan, "plan.json")
        return self.zip_values[endpoint]

    def add_authorization(self, record):
        artifact = {
            "id": 2 + len(self.artifacts),
            "name": "ci-manual-authorization-123",
            "size_in_bytes": 2000,
            "workflow_run": {"id": record.get("controller_run_id", 456)},
        }
        self.artifacts.append(artifact)
        self.zip_values[f"repos/{REPO}/actions/artifacts/{artifact['id']}/zip"] = (
            archive(record)
        )
        return artifact


class ManualCITests(unittest.TestCase):
    def setUp(self):
        self.github = FakeGitHub()
        self.environment = patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_SHA": MERGE,
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.event = {"number": 42, "pull_request": {"head": {"sha": HEAD}}}

    def authorize(self):
        event = {
            "action": "created",
            "repository": {"full_name": REPO},
            "sender": {"login": "maintainer"},
            "issue": {"number": 42, "pull_request": {}},
            "comment": copy.deepcopy(self.github.comment),
        }
        event["issue"]["pull_request"] = {"url": "https://api.github.com/pr"}
        with patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "issue_comment",
                "GITHUB_RUN_ID": "456",
                "GITHUB_SHA": TRUSTED,
            },
        ):
            return manual_ci.authorize(self.github, event)

    def resolve(self):
        return manual_ci.resolve(self.github, self.event, source=MERGE, base=BASE)

    def bootstrap(self, scope="full"):
        self.github.comment["body"] = f"/ci run {scope} {HEAD} {BASE} {MERGE}"
        self.github.comments = [self.github.comment]

    def test_command_parser_whitelists_scopes_and_full(self):
        self.assertEqual(
            manual_ci.parse_command("/ci run"),
            {"scopes": ["affected"], "full": False, "pinned": None},
        )
        self.assertEqual(manual_ci.parse_command("/ci run full")["full"], True)
        self.assertEqual(
            manual_ci.parse_command("/ci run models storage")["scopes"],
            ["models", "storage"],
        )
        self.assertEqual(
            manual_ci.parse_command(f"/ci run affected {HEAD} {BASE} {MERGE}")[
                "pinned"
            ],
            [HEAD, BASE, MERGE],
        )

    def test_command_parser_rejects_injection_edits_and_ambiguous_requests(self):
        for body in (
            "/ci run; whoami",
            "/ci run $(id)",
            "/ci run `id`",
            "/ci run\nfull",
            " /ci run",
            "/ci run ",
            "/ci  run",
            "/ci run full models",
            "/ci run arbitrary-target",
            "/ci run models models",
            f"/ci run full {HEAD}",
            f"/ci run models {HEAD} {BASE} {MERGE}",
        ):
            with self.subTest(body=body):
                self.assertIsNone(manual_ci.parse_command(body))

    def test_authorization_freezes_source_plan_actor_and_run(self):
        record = self.authorize()
        self.assertEqual(record["source_sha"], MERGE)
        self.assertEqual(record["head_sha"], HEAD)
        self.assertEqual(record["base_sha"], BASE)
        self.assertEqual(record["pr_run_id"], 123)
        self.assertEqual(record["controller_run_id"], 456)
        self.assertEqual(record["plan_sha256"], self.github.plan["plan_sha256"])
        self.assertEqual(record["scopes"], ["affected"])
        self.assertEqual(self.github.posts, [])

    def test_only_current_write_maintain_admin_permissions_authorize(self):
        for permission in ("write", "maintain", "admin"):
            self.github.permission = permission
            self.assertIsNotNone(self.authorize())
        for permission in ("read", "triage", "none"):
            self.github.permission = permission
            with self.assertRaisesRegex(manual_ci.ManualCIError, "Only repository"):
                self.authorize()

    def test_non_created_or_spoofed_comment_events_never_reach_authorization(self):
        event = {
            "action": "created",
            "repository": {"full_name": REPO},
            "sender": {"login": "maintainer"},
            "issue": {"number": 42, "pull_request": {"url": "pr-url"}},
            "comment": copy.deepcopy(self.github.comment),
        }
        variants = [
            {"action": "edited"},
            {"action": "deleted"},
            {"sender": {"login": "attacker"}},
            {"repository": {"full_name": "attacker/repo"}},
            {"issue": {"number": 42}},
        ]
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "issue_comment"}):
            for changes in variants:
                self.assertIsNone(
                    manual_ci.authorize(self.github, {**event, **changes})
                )
        self.assertEqual(self.github.calls, [])

    def test_edited_comment_cannot_authorize(self):
        self.github.comment["updated_at"] = "2026-10-18T10:02:00Z"
        with self.assertRaisesRegex(manual_ci.ManualCIError, "unedited"):
            self.authorize()

    def test_wrong_issue_or_changed_source_cannot_authorize(self):
        self.github.comment["issue_url"] = (
            f"https://api.github.com/repos/{REPO}/issues/43"
        )
        with self.assertRaises(manual_ci.ManualCIError):
            self.authorize()
        self.github.comment["issue_url"] = (
            f"https://api.github.com/repos/{REPO}/issues/42"
        )
        self.github.pull["merge_commit_sha"] = "e" * 40
        with self.assertRaisesRegex(manual_ci.ManualCIError, "No current PR CI plan"):
            self.authorize()

    def test_current_plan_digest_must_be_valid(self):
        self.github.plan["plan_sha256"] = "e" * 64
        with self.assertRaisesRegex(manual_ci.ManualCIError, "No current PR CI plan"):
            self.authorize()

    def test_run_must_belong_to_this_pr_and_predate_the_command(self):
        self.github.pr_run["pull_requests"] = [{"number": 43}]
        with self.assertRaises(manual_ci.ManualCIError):
            self.authorize()
        self.github.pr_run["pull_requests"] = [{"number": 42}]
        self.github.pr_run["created_at"] = "2026-10-18T10:02:00Z"
        with self.assertRaises(manual_ci.ManualCIError):
            self.authorize()

    def test_fork_run_with_empty_pr_metadata_uses_repository_branch_and_merge_plan(
        self,
    ):
        self.github.pr_run["pull_requests"] = []
        self.assertEqual(self.authorize()["pr_run_id"], 123)
        self.github.pr_run["head_repository"] = {"full_name": "other/fork"}
        with self.assertRaises(manual_ci.ManualCIError):
            self.authorize()

    def test_pending_latest_plan_cannot_fall_back_to_a_completed_older_run(self):
        self.github.pr_runs.append(
            {**self.github.pr_run, "id": 124, "created_at": "2026-10-18T10:00:30Z"}
        )
        with self.assertRaisesRegex(manual_ci.ManualCIError, "No current PR CI plan"):
            self.authorize()
        self.assertFalse(
            any("/runs/123/artifacts?" in call for call in self.github.calls)
        )

    def test_comment_before_latest_run_cannot_authorize_an_older_run(self):
        self.github.pr_runs.append(
            {**self.github.pr_run, "id": 124, "created_at": "2026-10-18T10:02:00Z"}
        )
        with self.assertRaisesRegex(manual_ci.ManualCIError, "No current PR CI plan"):
            self.authorize()
        self.assertFalse(any("/artifacts?" in call for call in self.github.calls))

    def test_no_authorization_runs_only_automatic_scope(self):
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_trusted_mergify_queue_runs_affected_qualification_without_reapproval(self):
        self.github.pull["user"] = {
            "id": 37929162,
            "login": "mergify[bot]",
            "type": "Bot",
        }
        self.github.pull["head"].update(
            ref="mergify/merge-queue/main/pr-42", repo={"full_name": REPO}
        )
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )

    def test_queue_prefix_does_not_authorize_humans_forks_or_other_bots(self):
        self.github.pull["user"] = {
            "id": 37929162,
            "login": "mergify[bot]",
            "type": "Bot",
        }
        self.github.pull["head"].update(
            ref="mergify/merge-queue/main/pr-42", repo={"full_name": REPO}
        )
        original = copy.deepcopy(self.github.pull)
        variants = [
            ("user", {"id": 123, "login": "maintainer", "type": "User"}),
            ("user", {"id": 123, "login": "mergify[bot]", "type": "Bot"}),
            ("user", {"id": 37929162, "login": "mergify[bot]", "type": "User"}),
            ("head", {**original["head"], "repo": {"full_name": "attacker/fork"}}),
            ("head", {**original["head"], "ref": "feature"}),
        ]
        for field, value in variants:
            with self.subTest(field=field, value=value):
                self.github.pull = {**copy.deepcopy(original), field: value}
                self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_trusted_artifact_authorizes_only_the_original_pr_run(self):
        record = self.authorize()
        self.github.add_authorization(record)
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )
        record["pr_run_id"] = 124
        self.github.zip_values[f"repos/{REPO}/actions/artifacts/2/zip"] = archive(
            record
        )
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_multiple_valid_requests_accumulate_scopes_for_the_same_revision(self):
        self.github.comment["body"] = "/ci run models"
        models = self.authorize()
        self.github.add_authorization(models)
        product_comment = {**self.github.comment, "id": 100, "body": "/ci run product"}
        self.github.additional_comments[100] = product_comment
        product = {
            **models,
            "comment_id": 100,
            "controller_run_id": 457,
            "scopes": ["product"],
        }
        self.github.add_authorization(product)
        self.assertEqual(
            self.resolve(),
            {"authorized": True, "scopes": ["models", "product"], "full": False},
        )

    def test_full_authorization_accumulates_without_dropping_existing_scopes(self):
        self.github.comment["body"] = "/ci run models"
        models = self.authorize()
        self.github.add_authorization(models)
        self.github.additional_comments[101] = {
            **self.github.comment,
            "id": 101,
            "body": "/ci run full",
        }
        full = {
            **models,
            "comment_id": 101,
            "controller_run_id": 458,
            "scopes": [],
            "full": True,
        }
        self.github.add_authorization(full)
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["models"], "full": True}
        )

    def test_untrusted_controller_cannot_forge_an_authorization_artifact(self):
        record = self.authorize()
        self.github.add_authorization(record)
        original = copy.deepcopy(self.github.controller)
        for key, value in (
            ("event", "pull_request"),
            ("path", ".github/workflows/fake.yml"),
            ("workflow_id", 11),
            ("head_branch", "feature"),
            ("head_sha", "f" * 40),
            ("head_repository", {"full_name": "attacker/fork"}),
            ("actor", {"login": "attacker"}),
        ):
            with self.subTest(key=key):
                self.github.controller = {**original, key: value}
                self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)
        self.github.controller = original
        self.github.trusted_ancestor = "f" * 40
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_authorize_job_must_have_completed_successfully(self):
        self.github.add_authorization(self.authorize())
        for status, conclusion in (
            ("in_progress", None),
            ("completed", "failure"),
            ("completed", "skipped"),
        ):
            self.github.authorize_job.update(status=status, conclusion=conclusion)
            self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_comment_edit_or_permission_revocation_invalidates_artifact(self):
        self.github.add_authorization(self.authorize())
        self.github.comment["updated_at"] = "2026-10-18T10:02:00Z"
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)
        self.github.comment["updated_at"] = REQUESTED
        self.github.permission = "read"
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_new_head_base_or_merge_invalidates_authorization(self):
        self.github.add_authorization(self.authorize())
        original = copy.deepcopy(self.github.pull)
        for field in ("head", "base", "merge_commit_sha"):
            self.github.pull = copy.deepcopy(original)
            self.github.base_ref["object"]["sha"] = BASE
            self.github.merge_parents = [{"sha": BASE}, {"sha": HEAD}]
            if field == "merge_commit_sha":
                self.github.pull[field] = "f" * 40
            else:
                self.github.pull[field]["sha"] = "f" * 40
                if field == "base":
                    self.github.base_ref["object"]["sha"] = "f" * 40
                    self.github.merge_parents[0]["sha"] = "f" * 40
                else:
                    self.github.merge_parents[1]["sha"] = "f" * 40
            self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_stale_pr_base_metadata_does_not_reject_the_actual_merge_revision(self):
        self.github.pull["base"]["sha"] = "e" * 40
        record = self.authorize()
        self.assertEqual(record["base_sha"], BASE)
        self.github.add_authorization(record)
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )
        manual_ci.rerun(self.github, record)
        self.assertEqual(self.github.posts, [f"repos/{REPO}/actions/runs/123/rerun"])
        self.github.artifacts = []
        self.bootstrap()
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": [], "full": True}
        )

    def test_advanced_main_requires_a_regenerated_merge_before_authorization(self):
        record = self.authorize()
        self.github.base_ref["object"]["sha"] = "e" * 40
        for operation in (
            self.authorize,
            self.resolve,
            lambda: manual_ci.rerun(self.github, record),
        ):
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(
                    manual_ci.ManualCIError, "merge revision"
                ) as error,
            ):
                operation()
            self.assertEqual(error.exception.reason, "revision")
        self.assertEqual(self.github.posts, [])

    def test_current_base_ref_must_be_main_and_a_valid_commit(self):
        original = copy.deepcopy(self.github.base_ref)
        for changes in (
            {"ref": "refs/heads/feature"},
            {"object": {"type": "tag", "sha": BASE}},
            {"object": {"type": "commit", "sha": "main"}},
            {"object": {"type": "commit", "sha": None}},
        ):
            with self.subTest(changes=changes):
                self.github.base_ref = {**original, **changes}
                with self.assertRaises(manual_ci.ManualCIError):
                    self.authorize()

    def test_merge_parents_must_be_exactly_the_live_base_then_current_head(self):
        for parents in (
            [],
            [{"sha": BASE}],
            [{"sha": HEAD}, {"sha": BASE}],
            [{"sha": "e" * 40}, {"sha": HEAD}],
            [{"sha": BASE}, {"sha": "e" * 40}],
            [{"sha": BASE}, {"sha": HEAD}, {"sha": "e" * 40}],
            [{"sha": BASE}, {"sha": HEAD}, None],
            None,
        ):
            with self.subTest(parents=parents):
                self.github.merge_parents = parents
                with self.assertRaises(manual_ci.ManualCIError):
                    self.authorize()

    def test_bootstrap_requires_full_explicit_current_tuple(self):
        self.bootstrap()
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": [], "full": True}
        )
        self.bootstrap("affected")
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )
        self.github.comment["body"] = "/ci run full"
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_bootstrap_rejects_edited_old_unauthorized_or_stale_commands(self):
        self.bootstrap()
        original = copy.deepcopy(self.github.comment)
        for changes in (
            {"updated_at": "2026-10-18T10:02:00Z"},
            {
                "created_at": "2026-10-18T09:00:00Z",
                "updated_at": "2026-10-18T09:00:00Z",
            },
            {"body": f"/ci run full {HEAD} {BASE} {'f' * 40}"},
        ):
            self.github.comments = [{**original, **changes}]
            self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)
        self.github.comments = [original]
        self.github.permission = "triage"
        self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)

    def test_non_pr_entrypoints_do_not_query_or_consume_authorization(self):
        for event_name in ("push", "workflow_dispatch", "issue_comment", "schedule"):
            with patch.dict(os.environ, {"GITHUB_EVENT_NAME": event_name}):
                self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)
        self.assertEqual(self.github.calls, [])

    def test_rerun_uses_original_pr_run_without_publishing_statuses(self):
        record = self.authorize()
        manual_ci.rerun(self.github, record)
        self.assertEqual(self.github.posts, [f"repos/{REPO}/actions/runs/123/rerun"])

    def test_rerun_rechecks_revision_and_comment_permission(self):
        record = self.authorize()
        self.github.pull["head"]["sha"] = "f" * 40
        with self.assertRaisesRegex(manual_ci.ManualCIError, "changed"):
            manual_ci.rerun(self.github, record)
        self.assertEqual(self.github.posts, [])

    def test_rerun_wait_is_bounded(self):
        record = self.authorize()
        self.github.pr_run["status"] = "in_progress"
        with self.assertRaisesRegex(manual_ci.ManualCIError, "still running"):
            manual_ci.rerun(self.github, record, wait_seconds=0)
        self.assertEqual(self.github.posts, [])

    def test_artifacts_are_read_as_bounded_json_without_extraction(self):
        artifact = self.github.add_authorization({"valid": True})
        endpoint = f"repos/{REPO}/actions/artifacts/2/zip"
        for raw in (
            b"not a zip",
            archive({"valid": True}, "../authorization.json"),
            archive(["not an object"]),
        ):
            self.github.zip_values[endpoint] = raw
            with self.assertRaises(manual_ci.ManualCIError):
                manual_ci.read_artifact(
                    self.github,
                    REPO,
                    artifact,
                    "authorization.json",
                    manual_ci.AUTHORIZATION_LIMIT,
                )
        artifact["size_in_bytes"] = manual_ci.AUTHORIZATION_LIMIT + 1
        with self.assertRaises(manual_ci.ManualCIError):
            manual_ci.read_artifact(
                self.github,
                REPO,
                artifact,
                "authorization.json",
                manual_ci.AUTHORIZATION_LIMIT,
            )


class ManualCIWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.load(
            (ROOT / manual_ci.CONTROLLER).read_text(), Loader=yaml.BaseLoader
        )

    def test_only_created_comments_trigger_trusted_code(self):
        self.assertEqual(self.workflow["on"], {"issue_comment": {"types": ["created"]}})
        for job in self.workflow["jobs"].values():
            self.assertEqual(job["runs-on"], "ubuntu-24.04")
            self.assertNotIn("secrets", job)
            checkout = next(
                step
                for step in job["steps"]
                if step.get("uses", "").startswith("actions/checkout@")
            )
            self.assertEqual(checkout["with"]["ref"], "${{ github.sha }}")
            self.assertEqual(checkout["with"]["persist-credentials"], "false")

    def test_actions_write_is_confined_to_rerun_after_successful_authorization(self):
        self.assertNotIn("write", self.workflow["permissions"].values())
        authorize = self.workflow["jobs"]["authorize"]
        self.assertEqual(
            authorize["permissions"],
            {
                "contents": "read",
                "actions": "read",
                "issues": "write",
                "pull-requests": "write",
            },
        )
        rerun = self.workflow["jobs"]["rerun"]
        self.assertEqual(rerun["needs"], "authorize")
        self.assertEqual(
            rerun["permissions"],
            {
                "contents": "read",
                "actions": "write",
                "issues": "write",
                "pull-requests": "write",
            },
        )
        self.assertEqual(rerun["if"], "needs.authorize.outputs.authorized == 'true'")

    def test_pr_gate_cannot_be_skipped_by_an_unrelated_label(self):
        workflow = yaml.load(
            (ROOT / ".github/workflows/pr.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertNotIn("labeled", workflow["on"]["pull_request"]["types"])
        self.assertNotIn("unlabeled", workflow["on"]["pull_request"]["types"])
        self.assertEqual(workflow["jobs"]["pr-gate"]["if"], "always()")

    def test_artifact_feedback_fallback_cannot_replace_handler_rejections(self):
        for job in self.workflow["jobs"].values():
            fallback = next(
                step
                for step in job["steps"]
                if "manual_ci.py feedback" in step.get("run", "")
            )
            self.assertEqual(
                fallback["if"],
                "failure() && steps.authorization-artifact.outcome == 'failure'",
            )
            self.assertIn(
                '--event "$GITHUB_EVENT_PATH" --reason controller',
                " ".join(fallback["run"].split()),
            )


if __name__ == "__main__":
    unittest.main()
