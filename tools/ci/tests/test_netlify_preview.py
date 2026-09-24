import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

SCRIPT = Path(__file__).resolve().parents[1] / "netlify_preview.py"
SPEC = importlib.util.spec_from_file_location("netlify_preview", SCRIPT)
assert SPEC and SPEC.loader
preview = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preview
SPEC.loader.exec_module(preview)
REPO, SHA = "vllm-project/semantic-router", "a" * 40


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root / "site"
        self.directory.mkdir()
        (self.directory / "index.html").write_text("<h1>preview</h1>")
        self.env = patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "issue_comment",
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_RUN_ID": "123",
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_OUTPUT": str(self.root / "output"),
                "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
                "NETLIFY_AUTH_TOKEN": "private-token",
                "NETLIFY_SITE_ID": "private-site",
                "PR_NUMBER": "42",
                "SHA": SHA,
                "BUILD_RESULT": "success",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.event = {
            "action": "created",
            "sender": {"login": "maintainer"},
            "repository": {"full_name": REPO},
            "comment": {"body": "/netlify", "user": {"login": "maintainer"}},
            "issue": {"number": 42, "pull_request": {"url": "pr-url"}},
        }
        self.pull = {
            "state": "open",
            "base": {"ref": "main", "repo": {"full_name": REPO}},
            "head": {"sha": SHA, "repo": {"full_name": "contributor/fork"}},
        }
        self.permission = "write"
        self.pages = {1: []}
        self.run_state = "in_progress"
        self.posts = []
        self.github = Mock()
        self.github.request.side_effect = self.github_request
        self.stdout = io.StringIO()
        redirect = redirect_stdout(self.stdout)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def github_request(self, endpoint, *, method="GET", payload=None):
        if method == "POST":
            self.posts.append((endpoint, payload))
            return payload
        if "/collaborators/" in endpoint:
            return {"permission": self.permission}
        if endpoint.endswith("/pulls/42"):
            return self.pull
        if "/statuses?" in endpoint:
            return self.pages[int(parse_qs(urlparse(endpoint).query)["page"][0])]
        if "/actions/runs/" in endpoint:
            return {"status": self.run_state}
        self.fail(f"Unexpected API endpoint: {endpoint}")

    def authorize(self):
        return preview.authorize(self.github, self.event)

    def entry(self, state, *, number=42, run=123):
        return {
            "context": f"netlify-preview/PR-{number}",
            "state": state,
            "target_url": f"https://github.com/{REPO}/actions/runs/{run}",
        }

    def test_authorizes_write_maintain_admin_and_pins_fork_head(self):
        for permission in ("write", "maintain", "admin"):
            with self.subTest(permission=permission):
                self.permission = permission
                self.assertEqual(
                    self.authorize(),
                    {
                        "build": "true",
                        "sha": SHA,
                        "number": 42,
                        "repository": "contributor/fork",
                    },
                )
                endpoint, posted = self.posts[-1]
                self.assertEqual(endpoint, f"repos/{REPO}/statuses/{SHA}")
                self.assertEqual(posted["state"], "pending")
                self.assertEqual(posted["context"], "netlify-preview/PR-42")

    def test_exact_created_pr_comment_is_required(self):
        variants = [
            ("action", "edited"),
            ("comment", {"body": "please /netlify"}),
            ("comment", {"body": "/netlify\nextra"}),
            ("comment", {"body": " /netlify"}),
            ("comment", {"body": "/netlify\n"}),
            ("issue", {"number": 42}),
            ("sender", {"login": "different-user"}),
            ("repository", {"full_name": "somewhere/else"}),
        ]
        original = copy.deepcopy(self.event)
        for key, value in variants:
            with self.subTest(key=key, value=value):
                self.event = copy.deepcopy(original)
                if key == "comment":
                    self.event[key].update(value)
                else:
                    self.event[key] = value
                self.assertEqual(self.authorize(), {"build": "false"})
        self.github.request.assert_not_called()

    def test_push_pull_request_and_dispatch_events_do_not_build(self):
        for name in (
            "push",
            "pull_request",
            "pull_request_target",
            "workflow_dispatch",
        ):
            with self.subTest(name=name), patch.dict(
                os.environ, {"GITHUB_EVENT_NAME": name}
            ):
                self.assertEqual(self.authorize(), {"build": "false"})
        self.github.request.assert_not_called()

    def test_read_triage_and_bots_have_no_permission_bypass(self):
        for actor in ("maintainer", "github-actions[bot]"):
            for permission in ("read", "triage", "none"):
                with self.subTest(actor=actor, permission=permission):
                    self.event["sender"]["login"] = actor
                    self.event["comment"]["user"]["login"] = actor
                    self.event["comment"]["author_association"] = "OWNER"
                    self.permission = permission
                    self.assertEqual(self.authorize(), {"build": "false"})
        self.assertEqual(self.posts, [])
        self.assertEqual(self.github.request.call_count, 6)

    def test_closed_or_wrong_base_pr_is_ignored(self):
        original = copy.deepcopy(self.pull)
        for state, branch, repo in (
            ("closed", "main", REPO),
            ("open", "release", REPO),
            ("open", "main", "other/repo"),
        ):
            self.pull = copy.deepcopy(original)
            self.pull["state"], self.pull["base"]["ref"] = state, branch
            self.pull["base"]["repo"]["full_name"] = repo
            self.assertEqual(self.authorize(), {"build": "false"})
        self.assertEqual(self.posts, [])

    def test_missing_secrets_fail_before_pending_status(self):
        with patch.dict(os.environ, {"NETLIFY_AUTH_TOKEN": ""}), self.assertRaisesRegex(
            preview.PreviewError, "Configure NETLIFY_AUTH_TOKEN"
        ):
            self.authorize()
        self.assertEqual(self.posts, [])
        self.assertEqual((self.root / "output").read_text(), "build=false\n")

    def test_deleted_fork_head_is_rejected(self):
        self.pull["head"]["repo"] = None
        with self.assertRaisesRegex(preview.PreviewError, "head is unavailable"):
            self.authorize()
        self.assertEqual(self.posts, [])

    def test_latest_success_or_active_pending_skips_build(self):
        for state in ("success", "pending"):
            with self.subTest(state=state):
                self.pages[1] = [self.entry(state)]
                self.assertEqual(self.authorize(), {"build": "false"})
        self.assertEqual(self.posts, [])

    def test_status_pagination_finds_older_success(self):
        self.pages[1] = [self.entry("success", number=100)] * 100
        self.pages[2] = [self.entry("success")]
        self.assertEqual(self.authorize(), {"build": "false"})
        self.assertTrue(
            any("page=2" in call.args[0] for call in self.github.request.call_args_list)
        )

    def test_latest_failure_can_retry_even_with_older_success(self):
        self.pages[1] = [self.entry("failure"), self.entry("success")]
        self.assertEqual(self.authorize()["build"], "true")

    def test_finished_run_pending_status_can_retry(self):
        self.pages[1] = [self.entry("pending", run=100)]
        self.run_state = "completed"
        self.assertEqual(self.authorize()["build"], "true")

    def test_cancelled_attempt_can_retry_in_the_same_workflow_run(self):
        self.pages[1] = [self.entry("pending", run=123)]
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            self.assertEqual(self.authorize()["build"], "true")

    def test_unknown_pending_run_is_not_retried(self):
        entry = self.entry("pending")
        entry["target_url"] = "https://example.com/unknown"
        self.pages[1] = [entry]
        self.assertEqual(self.authorize(), {"build": "false"})

    def test_new_sha_and_other_pr_status_do_not_suppress_build(self):
        new_sha = "b" * 40
        self.pull["head"]["sha"] = new_sha
        self.pages[1] = [self.entry("success", number=100)]
        self.assertEqual(self.authorize()["sha"], new_sha)
        self.assertIn(
            f"/{new_sha}/statuses?", self.github.request.call_args_list[-2].args[0]
        )

    def test_static_manifest_and_duplicate_hashes(self):
        (self.directory / "same.html").write_bytes(
            (self.directory / "index.html").read_bytes()
        )
        files, by_digest = preview.static_files(self.directory)
        self.assertEqual(set(files), {"/index.html", "/same.html"})
        self.assertEqual(len(by_digest), 1)

    def test_static_build_requires_index_and_rejects_symlinks(self):
        (self.directory / "index.html").unlink()
        with self.assertRaisesRegex(preview.PreviewError, "index.html"):
            preview.static_files(self.directory)
        outside = self.root / "outside"
        outside.write_text("secret")
        (self.directory / "index.html").symlink_to(outside)
        with self.assertRaisesRegex(preview.PreviewError, "symlink"):
            preview.static_files(self.directory)

    def test_static_build_rejects_directory_symlinks_and_invalid_names(self):
        link = self.directory / "linked-directory"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(preview.PreviewError, "symlink"):
            preview.static_files(self.directory)
        link.unlink()
        (self.directory / "bad?name").write_text("bad")
        with self.assertRaisesRegex(preview.PreviewError, "unsupported file name"):
            preview.static_files(self.directory)

    def test_digest_api_creates_only_draft_and_uploads_required_static_files(self):
        (self.directory / "name with spaces.js").write_text("alert('preview')")
        files, _ = preview.static_files(self.directory)
        netlify = Mock()
        netlify.request.side_effect = [
            {
                "id": "deploy-id",
                "state": "uploading",
                "required": [files["/name with spaces.js"]],
            },
            {},
            {
                "id": "deploy-id",
                "state": "ready",
                "deploy_ssl_url": "https://draft.netlify.app",
            },
        ]
        self.assertEqual(
            preview.deploy_static(netlify, "site-id", self.directory, 42, SHA),
            "https://draft.netlify.app",
        )
        create, upload, poll = netlify.request.call_args_list
        self.assertEqual(create.kwargs["method"], "POST")
        self.assertEqual(
            create.kwargs["payload"],
            {
                "files": files,
                "draft": True,
                "branch": "deploy-preview-42",
            },
        )
        self.assertEqual(
            parse_qs(urlparse(create.args[0]).query)["title"], [f"PR #42 ({SHA[:12]})"]
        )
        self.assertEqual(
            upload.args[0], "deploys/deploy-id/files/name%20with%20spaces.js"
        )
        self.assertEqual(upload.kwargs["data"], b"alert('preview')")
        self.assertEqual(poll.args[0], "deploys/deploy-id")

    def test_create_deploy_is_never_retried(self):
        netlify = Mock()
        netlify.request.side_effect = preview.PreviewError("API unavailable")
        with self.assertRaises(preview.PreviewError):
            preview.deploy_static(netlify, "site-id", self.directory, 42, SHA)
        self.assertEqual(netlify.request.call_count, 1)

    def test_processing_failure_and_timeout_are_bounded(self):
        netlify = Mock()
        netlify.request.return_value = {"id": "deploy-id", "state": "processing"}
        with patch.object(preview.time, "sleep"), self.assertRaisesRegex(
            preview.PreviewError, "Timed out"
        ):
            preview.wait_ready(netlify, {"id": "deploy-id"}, attempts=3)
        self.assertEqual(netlify.request.call_count, 3)
        with self.assertRaisesRegex(preview.PreviewError, "could not process"):
            preview.wait_ready(netlify, {"id": "deploy-id", "state": "error"})

    def test_build_failure_never_reaches_netlify(self):
        netlify = Mock()
        with patch.dict(
            os.environ, {"BUILD_RESULT": "failure"}
        ), self.assertRaisesRegex(
            preview.PreviewError, "Website build did not succeed"
        ):
            preview.publish(self.github, self.directory, netlify)
        netlify.request.assert_not_called()
        self.assertEqual(self.posts[-1][1]["state"], "failure")

    def test_closed_or_stale_pr_never_reaches_netlify(self):
        for changed in ("closed", "new-sha"):
            with self.subTest(changed=changed):
                self.pull["state"] = "closed" if changed == "closed" else "open"
                self.pull["head"]["sha"] = "b" * 40
                netlify = Mock()
                with self.assertRaisesRegex(
                    preview.PreviewError, "PR closed or head changed"
                ):
                    preview.publish(self.github, self.directory, netlify)
                netlify.request.assert_not_called()
                self.assertEqual(self.posts[-1][1]["state"], "error")

    def test_publish_records_failure_after_api_error(self):
        netlify = Mock()
        netlify.request.side_effect = preview.PreviewError(
            "Netlify API returned HTTP 500."
        )
        with self.assertRaises(preview.PreviewError):
            preview.publish(self.github, self.directory, netlify)
        self.assertEqual(self.posts[-1][1]["state"], "failure")

    def test_publish_records_ready_url_in_status_and_summary(self):
        netlify = Mock()
        netlify.request.return_value = {
            "id": "deploy-id",
            "state": "ready",
            "required": [],
            "deploy_ssl_url": "https://preview.netlify.app",
        }
        preview.publish(self.github, self.directory, netlify)
        self.assertEqual(self.posts[-1][1]["state"], "success")
        self.assertEqual(self.posts[-1][1]["target_url"], "https://preview.netlify.app")
        self.assertIn(
            "https://preview.netlify.app", (self.root / "summary").read_text()
        )


class ClientTests(unittest.TestCase):
    def test_github_errors_do_not_echo_command_output(self):
        result = subprocess.CompletedProcess([], 1, "secret", "private-token")
        with patch.object(
            preview.subprocess, "run", return_value=result
        ), self.assertRaises(preview.PreviewError) as raised:
            preview.GitHubClient().request("repos/example/repo")
        self.assertNotIn("private-token", str(raised.exception))

    def test_netlify_errors_hide_response_body_and_transport_details(self):
        client = preview.NetlifyClient("private-token")
        for error in (
            HTTPError(
                "private-url", 403, "private-token", {}, io.BytesIO(b"private-token")
            ),
            URLError("private-token"),
        ):
            with self.subTest(error=type(error).__name__):
                client.opener = Mock()
                client.opener.open.side_effect = error
                with self.assertRaises(preview.PreviewError) as raised:
                    client.request(
                        "sites/site/deploys", method="POST", payload={"draft": True}
                    )
                self.assertNotIn("private-token", str(raised.exception))
                self.assertEqual(client.opener.open.call_count, 1)

    def test_netlify_requests_use_fixed_https_host_and_bearer_auth(self):
        client = preview.NetlifyClient("private-token")
        client.opener = Mock()
        client.opener.open.return_value.__enter__ = Mock(
            return_value=io.BytesIO(b'{"id":"deploy-id"}')
        )
        client.opener.open.return_value.__exit__ = Mock(return_value=False)
        client.request("sites/site/deploys", method="POST", payload={"draft": True})
        request = client.opener.open.call_args.args[0]
        self.assertEqual(
            request.full_url, "https://api.netlify.com/api/v1/sites/site/deploys"
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer private-token")
        self.assertEqual(json.loads(request.data), {"draft": True})


if __name__ == "__main__":
    unittest.main()
