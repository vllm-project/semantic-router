"""Command feedback never grants permission or changes a rerun's outcome."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

import manual_ci  # noqa: E402
from manual_ci_feedback import feedback_for_event, warning  # noqa: E402
from test_manual_ci import REPO, TRUSTED, FakeGitHub  # noqa: E402


class FeedbackGitHub(FakeGitHub):
    def __init__(self):
        super().__init__()
        self.writes = []
        self.timeline = []
        self.reactions = set()
        self.reaction_items = []
        self.next_reaction_id = 1000
        self.deleted_reactions = []
        self.feedback_error = False
        self.lookup_error = False
        self.trigger_error = None
        self.delete_error = False

    def request(self, endpoint, *, method="GET"):
        if "/reactions?" in endpoint:
            return list(self.reaction_items)
        if "/reactions/" in endpoint and method == "DELETE":
            if self.delete_error:
                raise manual_ci.ManualCIError("SECRET deletion failure")
            identity = int(endpoint.rsplit("/", 1)[-1])
            self.deleted_reactions.append(identity)
            self.reaction_items = [
                item for item in self.reaction_items if item["id"] != identity
            ]
            self.reactions.discard("confused")
            return None
        if endpoint.endswith("/issues/comments/99") and self.lookup_error:
            self.lookup_error = False
            raise manual_ci.ManualCIError("SECRET lookup error")
        if endpoint.endswith("/rerun"):
            self.timeline.append("rerun")
            if self.trigger_error:
                raise self.trigger_error
        return super().request(endpoint, method=method)

    def write(self, endpoint, payload, *, method="POST"):
        if self.feedback_error:
            raise manual_ci.ManualCIError("SECRET feedback error")
        self.writes.append((endpoint, copy.deepcopy(payload), method))
        if endpoint.endswith("/reactions"):
            self.reactions.add(payload["content"])
            self.timeline.append(payload["content"])
            if not any(
                item["content"] == payload["content"]
                and item["user"]["login"] == "github-actions[bot]"
                for item in self.reaction_items
            ):
                self.reaction_items.append(
                    {
                        "id": self.next_reaction_id,
                        "content": payload["content"],
                        "user": {"login": "github-actions[bot]", "type": "Bot"},
                    }
                )
                self.next_reaction_id += 1
        elif method == "PATCH":
            for comment in self.comments:
                if endpoint.endswith(f"/comments/{comment['id']}"):
                    comment.update(payload)
                    return comment
            raise AssertionError(f"Unknown comment: {endpoint}")
        else:
            comment = {
                "id": 900 + len(self.comments),
                "user": {"login": "github-actions[bot]", "type": "Bot"},
                **payload,
            }
            self.comments.append(comment)
            return comment
        return {}


class ManualCIFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.client = FeedbackGitHub()
        environment = patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "issue_comment",
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_RUN_ID": "456",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_SHA": TRUSTED,
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.event = {
            "action": "created",
            "repository": {"full_name": REPO},
            "sender": {"login": "maintainer"},
            "issue": {"number": 42, "pull_request": {"url": "pr-url"}},
            "comment": copy.deepcopy(self.client.comment),
        }

    def test_valid_request_is_received_and_rocket_follows_actual_rerun(self):
        record = manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(self.client.reactions, {"eyes"})
        self.assertEqual(self.client.posts, [])
        manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertEqual(self.client.timeline, ["eyes", "rerun", "rocket"])
        self.assertEqual(self.client.comments, [])

    def test_permission_denial_explains_who_can_request_without_authorizing(self):
        self.client.permission = "read"
        with self.assertRaises(manual_ci.ManualCIError):
            manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(self.client.reactions, {"eyes", "confused"})
        body = self.client.comments[0]["body"]
        self.assertIn("Ask a committer with write, maintain or admin access", body)
        self.assertIn("post `/ci run`", body)
        self.assertEqual(self.client.posts, [])

    def test_invalid_command_gets_syntax_help_without_echoing_untrusted_text(self):
        for body in (
            "/ci run $(echo SECRET)",
            "/ci run missing-scope",
            "/ci run full models",
        ):
            with self.subTest(body=body):
                self.client.comment["body"] = body
                self.event["comment"]["body"] = body
                with self.assertRaises(manual_ci.ManualCIError):
                    manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(self.client.reactions, {"confused"})
        self.assertEqual(len(self.client.comments), 1)
        self.assertIn("`/ci run full`", self.client.comments[0]["body"])
        self.assertIn("`performance`", self.client.comments[0]["body"])
        self.assertNotIn("SECRET", str(self.client.writes))
        self.assertEqual(self.client.posts, [])

    def test_non_commands_and_spoofed_creation_events_get_no_feedback(self):
        variants = (
            {"action": "edited"},
            {"action": "deleted"},
            {"repository": {"full_name": "other/repository"}},
            {"sender": {"login": "someone-else"}},
            {"issue": {"number": 42}},
            {"comment": {**self.event["comment"], "body": "Please run CI"}},
        )
        for changes in variants:
            with self.subTest(changes=changes):
                self.assertIsNone(
                    manual_ci.authorize_with_feedback(
                        self.client, {**self.event, **changes}
                    )
                )
        self.assertEqual(self.client.writes, [])
        self.assertEqual(self.client.calls, [])

    def test_live_comment_must_belong_to_the_exact_pr_and_actor(self):
        for changes in (
            {"issue_url": f"https://api.github.com/repos/{REPO}/issues/43"},
            {"user": {"login": "someone-else"}},
        ):
            with self.subTest(changes=changes):
                self.client.comment = {**self.event["comment"], **changes}
                with self.assertRaises(manual_ci.ManualCIError):
                    manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(self.client.writes, [])

    def test_expired_authorization_reports_next_step_without_starting_ci(self):
        record = manual_ci.authorize_with_feedback(self.client, self.event)
        self.client.pull["head"]["sha"] = "f" * 40
        self.client.merge_parents[1]["sha"] = "f" * 40
        with self.assertRaises(manual_ci.ManualCIError):
            manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertNotIn("rocket", self.client.reactions)
        self.assertEqual(self.client.posts, [])
        self.assertIn(
            "revision or authorization changed", self.client.comments[0]["body"]
        )
        self.assertIn("new `/ci run`", self.client.comments[0]["body"])

    def test_stale_merge_reports_branch_update_instead_of_only_waiting_for_plan(self):
        self.client.base_ref["object"]["sha"] = "f" * 40
        with self.assertRaises(manual_ci.ManualCIError) as error:
            manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(error.exception.reason, "revision")
        self.assertEqual(self.client.posts, [])
        self.assertNotIn("rocket", self.client.reactions)
        self.assertIn(
            "Update the branch onto current `main` or resolve conflicts",
            self.client.comments[0]["body"],
        )

    def test_trigger_failure_or_timeout_is_not_reported_as_a_success_or_certainty(self):
        record = manual_ci.authorize_with_feedback(self.client, self.event)
        for error in (
            manual_ci.ManualCIError("SECRET API failure"),
            subprocess.TimeoutExpired("SECRET command", timeout=60),
        ):
            with self.subTest(error=type(error).__name__):
                self.client.trigger_error = error
                with self.assertRaises(type(error)):
                    manual_ci.rerun_with_feedback(self.client, record, self.event)
                self.assertNotIn("rocket", self.client.reactions)
        self.assertIn("could not confirm", self.client.comments[0]["body"])
        self.assertNotIn("SECRET", str(self.client.writes))

    def test_feedback_api_failure_cannot_invalidate_authorization_or_rerun(self):
        self.client.feedback_error = True
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            record = manual_ci.authorize_with_feedback(self.client, self.event)
            manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertEqual(self.client.posts, [f"repos/{REPO}/actions/runs/123/rerun"])
        self.assertIn("::warning::", output.getvalue())
        self.assertNotIn("SECRET", output.getvalue())

    def test_http_403_diagnostic_does_not_expose_api_content_or_change_authorization(
        self,
    ):
        self.client.write = manual_ci.GitHubClient().write
        output = io.StringIO()
        result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"message":"SECRET response body"}',
            stderr="gh: SECRET response and credential (HTTP 403)\n",
        )
        with (
            patch("manual_ci.subprocess.run", return_value=result),
            contextlib.redirect_stdout(output),
        ):
            record = manual_ci.authorize_with_feedback(self.client, self.event)
            manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertEqual(self.client.posts, [f"repos/{REPO}/actions/runs/123/rerun"])
        self.assertIn("(reaction; HTTP 403)", output.getvalue())
        self.assertNotIn("SECRET", output.getvalue())
        self.assertNotIn(REPO, output.getvalue())

    def test_malformed_success_response_has_no_public_payload_or_invented_status(self):
        self.client.write = manual_ci.GitHubClient().write
        result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="SECRET malformed JSON", stderr=""
        )
        output = io.StringIO()
        with (
            patch("manual_ci.subprocess.run", return_value=result),
            contextlib.redirect_stdout(output),
        ):
            record = manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(record["pr_run_id"], 123)
        self.assertIn("(reaction; HTTP status unavailable)", output.getvalue())
        self.assertNotIn("SECRET", output.getvalue())

    def test_transport_diagnostics_accept_only_a_bounded_http_error_suffix(self):
        for stderr, expected in (
            ("gh: SECRET (HTTP 403)\n", 403),
            ("gh: SECRET (HTTP 502)", 502),
            ("SECRET connection failed", None),
            ("SECRET (HTTP 4030)", None),
            ("SECRET (HTTP 999)", None),
            ("SECRET (HTTP 403)\nSECRET trailing payload", None),
        ):
            for raw in (False, True):
                with self.subTest(stderr=stderr, raw=raw):
                    result = subprocess.CompletedProcess(
                        args=[],
                        returncode=1,
                        stdout=b"SECRET body" if raw else "SECRET body",
                        stderr=stderr.encode() if raw else stderr,
                    )
                    client = manual_ci.GitHubClient()
                    with (
                        patch("manual_ci.subprocess.run", return_value=result),
                        self.assertRaises(manual_ci.ManualCIError) as error,
                    ):
                        if raw:
                            client.raw("SECRET endpoint")
                        else:
                            client.write("SECRET endpoint", {"body": "SECRET payload"})
                    self.assertEqual(error.exception.http_status, expected)
                    self.assertNotIn("SECRET", str(error.exception))
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        warning("comment", error.exception)
                    self.assertIn(
                        f"HTTP {expected}" if expected else "HTTP status unavailable",
                        output.getvalue(),
                    )
                    self.assertNotIn("SECRET", output.getvalue())

    def test_diagnostic_labels_and_status_metadata_cannot_inject_log_content(self):
        for status in ("403\nSECRET", True, 4030, -403, None):
            with self.subTest(status=status):
                error = manual_ci.ManualCIError("SECRET exception", http_status=status)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    warning("SECRET untrusted endpoint", error)
                self.assertIn("(feedback; HTTP status unavailable)", output.getvalue())
                self.assertNotIn("SECRET", output.getvalue())

    def test_failed_feedback_target_lookup_does_not_block_independent_authorization(
        self,
    ):
        self.client.lookup_error = True
        with contextlib.redirect_stdout(io.StringIO()):
            record = manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(record["pr_run_id"], 123)
        self.assertEqual(self.client.writes, [])

    def test_retry_updates_old_failure_in_place_with_the_latest_success(self):
        self.client.permission = "read"
        with self.assertRaises(manual_ci.ManualCIError):
            manual_ci.authorize_with_feedback(self.client, self.event)
        old_id = self.client.comments[0]["id"]
        self.client.permission = "write"
        record = manual_ci.authorize_with_feedback(self.client, self.event)
        manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertEqual(len(self.client.comments), 1)
        self.assertEqual(self.client.comments[0]["id"], old_id)
        self.assertIn("Latest result", self.client.comments[0]["body"])
        self.assertIn("CI rerun requested", self.client.comments[0]["body"])
        self.assertIn("does not mean tests passed", self.client.comments[0]["body"])
        self.assertNotIn("cannot authorize", self.client.comments[0]["body"])
        self.assertEqual(self.client.reactions, {"eyes", "rocket"})

    def test_success_only_removes_the_controllers_own_confused_reaction(self):
        record = manual_ci.authorize_with_feedback(self.client, self.event)
        self.client.reaction_items.extend(
            [
                {
                    "id": 81,
                    "content": "confused",
                    "user": {"login": "contributor", "type": "User"},
                },
                {
                    "id": 82,
                    "content": "confused",
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                },
                {
                    "id": 83,
                    "content": "eyes",
                    "user": {"login": "github-actions[bot]", "type": "Bot"},
                },
            ]
        )
        manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertEqual(self.client.deleted_reactions, [82])
        remaining = {item["id"] for item in self.client.reaction_items}
        self.assertIn(81, remaining)
        self.assertIn(83, remaining)

    def test_reaction_cleanup_failure_still_updates_the_successful_retry(self):
        self.client.permission = "read"
        with self.assertRaises(manual_ci.ManualCIError):
            manual_ci.authorize_with_feedback(self.client, self.event)
        self.client.permission = "write"
        self.client.delete_error = True
        record = manual_ci.authorize_with_feedback(self.client, self.event)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            manual_ci.rerun_with_feedback(self.client, record, self.event)
        self.assertIn("rocket", self.client.reactions)
        self.assertIn("Latest result", self.client.comments[0]["body"])
        self.assertIn("CI rerun requested", self.client.comments[0]["body"])
        self.assertNotIn("SECRET", output.getvalue())

    def test_cli_preserves_authorization_artifact_and_emits_feedback_at_each_stage(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            authorization = Path(directory) / "authorization.json"
            output = Path(directory) / "github-output"
            event.write_text(json.dumps(self.event))
            with (
                patch.dict(
                    os.environ,
                    {"GITHUB_EVENT_PATH": str(event), "GITHUB_OUTPUT": str(output)},
                ),
                patch.object(manual_ci, "GitHubClient", return_value=self.client),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "manual_ci.py",
                        "authorize",
                        "--event",
                        str(event),
                        "--output",
                        str(authorization),
                    ],
                ):
                    self.assertEqual(manual_ci.main(), 0)
                record = json.loads(authorization.read_text())
                self.assertEqual(record["comment_id"], self.event["comment"]["id"])
                self.assertIn("authorized=true", output.read_text())
                self.assertEqual(self.client.reactions, {"eyes"})
                with patch.object(
                    sys,
                    "argv",
                    ["manual_ci.py", "rerun", "--authorization", str(authorization)],
                ):
                    self.assertEqual(manual_ci.main(), 0)
                self.assertEqual(self.client.timeline, ["eyes", "rerun", "rocket"])

    def test_feedback_never_edits_another_users_marker_or_another_command(self):
        original = [
            {
                "id": 800,
                "user": {"login": "contributor", "type": "User"},
                "body": "<!-- semantic-router-ci-command:99 -->\nunchanged",
            },
            {
                "id": 801,
                "user": {"login": "github-actions[bot]", "type": "Bot"},
                "body": "<!-- semantic-router-ci-command:999 -->\nunchanged",
            },
        ]
        self.client.comments = copy.deepcopy(original)
        self.client.permission = "triage"
        with self.assertRaises(manual_ci.ManualCIError):
            manual_ci.authorize_with_feedback(self.client, self.event)
        self.assertEqual(self.client.comments[:2], original)
        self.assertEqual(len(self.client.comments), len(original) + 1)

    def test_rerun_feedback_record_must_match_original_event_target(self):
        record = manual_ci.authorize(self.client, self.event)
        for field, value in (
            ("repository", "other/repo"),
            ("pr_number", 43),
            ("comment_id", 100),
            ("actor", "other"),
        ):
            with self.subTest(field=field):
                self.assertIsNone(
                    feedback_for_event(
                        self.client, self.event, record={**record, field: value}
                    )
                )
        self.assertEqual(self.client.writes, [])

    def test_artifact_failure_cli_is_fixed_reason_and_best_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            event.write_text(json.dumps(self.event))
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "manual_ci.py",
                        "feedback",
                        "--event",
                        str(event),
                        "--reason",
                        "controller",
                    ],
                ),
                patch.object(manual_ci, "GitHubClient", return_value=self.client),
            ):
                self.assertEqual(manual_ci.main(), 0)
                self.assertIn(
                    "transfer its authorization evidence",
                    self.client.comments[0]["body"],
                )
                self.client.feedback_error = True
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(manual_ci.main(), 0)


if __name__ == "__main__":
    unittest.main()
