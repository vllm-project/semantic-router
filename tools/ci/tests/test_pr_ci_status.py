from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_ci import PR_REVISION_POLICY
from manual_ci_feedback import MESSAGES
from pr_ci_status import (
    APP_ID,
    CAPABILITIES,
    MARKER,
    PREFIX,
    main,
    reconcile,
)

REPO = "example/router"
HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


class Client:
    def __init__(self):
        self.run = {
            "id": 101,
            "workflow_id": 2,
            "event": "pull_request",
            "path": ".github/workflows/pr.yml",
            "head_sha": HEAD,
            "head_branch": "feature",
            "head_repository": {"full_name": "contributor/router"},
            "pull_requests": [],
            "created_at": "2026-10-19T00:00:00Z",
            "status": "completed",
            "conclusion": "success",
            "run_attempt": 1,
            "referenced_workflows": [
                {
                    "path": f"{REPO}/.github/workflows/ci.yml@{MERGE}",
                    "ref": "refs/pull/3/merge",
                    "sha": MERGE,
                }
            ],
        }
        self.pull = {
            "number": 3,
            "state": "open",
            "merge_commit_sha": MERGE,
            "head": {
                "sha": HEAD,
                "ref": "feature",
                "repo": {"full_name": "contributor/router"},
            },
            "base": {"ref": "main", "sha": BASE, "repo": {"full_name": REPO}},
        }
        self.plan = {
            "schema_version": 1,
            "source_sha": MERGE,
            "base_sha": BASE,
            "expected_verification_ids": ["quality"],
            "required_verification_ids": ["quality", "e2e.dashboard"],
            "deferred_verifications": ["e2e.dashboard"],
            "deferred_images": ["router", "dashboard"],
        }
        self.jobs = [
            {"name": name, "conclusion": "success"}
            for name in ["CI / Gate", "CI Execution", "Quality"]
        ]
        self.checks, self.comments, self.writes = [], [], []
        self.ready, self.missing_report, self.newer_run = True, False, False
        self.parents = [BASE, HEAD]
        self.main_sha, self.base_ancestor = BASE, BASE
        self.resolver_source = f'PR_REVISION_POLICY = "{PR_REVISION_POLICY}"\n'
        self.refresh_report()

    def refresh_report(self):
        unsigned = {k: v for k, v in self.plan.items() if k != "plan_sha256"}
        self.plan["plan_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.report = {
            key: copy.deepcopy(self.plan[key])
            for key in [
                "schema_version",
                "source_sha",
                "base_sha",
                "plan_sha256",
                "required_verification_ids",
                "deferred_verifications",
                "deferred_images",
            ]
        }
        self.report.update(
            run_id=101,
            run_attempt=self.run["run_attempt"],
            errors=[],
            status=(
                "awaiting-approval"
                if self.plan["deferred_verifications"] or self.plan["deferred_images"]
                else "success"
            ),
        )

    def request(self, endpoint, **kwargs):
        prefix = f"repos/{REPO}/"
        self.assert_prefix(endpoint, prefix)
        path = endpoint[len(prefix) :]
        if path == "git/ref/heads/main":
            return {
                "ref": "refs/heads/main",
                "object": {"type": "commit", "sha": self.main_sha},
            }
        if path.startswith(f"compare/{BASE}..."):
            return {"merge_base_commit": {"sha": self.base_ancestor}}
        if path.startswith("contents/tools/ci/manual_ci.py?"):
            content = self.resolver_source.encode()
            return {
                "type": "file",
                "encoding": "base64",
                "size": len(content),
                "content": base64.b64encode(content).decode(),
            }
        if path == "actions/runs/101":
            return copy.deepcopy(self.run)
        if path == "actions/workflows/pr.yml":
            return {"id": 2}
        if path.startswith("actions/workflows/pr.yml/runs?"):
            runs = [copy.deepcopy(self.run)]
            if self.newer_run:
                runs.append({**self.run, "id": 102})
            return {"workflow_runs": runs}
        if path.startswith("pulls?state=open&base=main&head=contributor%3Afeature"):
            return [copy.deepcopy(self.pull)]
        if path == "pulls/3":
            return copy.deepcopy(self.pull)
        if path.startswith(f"commits/{HEAD}/check-runs?"):
            checks = self.checks
            if "filter=latest" in path:
                # A rerun supersedes checks in earlier attempts' suites. PATCH
                # changes their verdict, but does not make them current again.
                checks = [
                    item
                    for item in checks
                    if item.get("suite_attempt", self.run["run_attempt"])
                    == self.run["run_attempt"]
                ]
            return {"check_runs": copy.deepcopy(checks)}
        if path.startswith("actions/runs/101/attempts/"):
            return {"jobs": copy.deepcopy(self.jobs)}
        if path.startswith("actions/runs/101/artifacts?"):
            gate = "name=ci-gate" in path
            if gate and self.missing_report:
                return {"artifacts": []}
            return {
                "artifacts": [
                    {
                        "id": 2 if gate else 1,
                        "name": "ci-gate" if gate else "ci-plan",
                        "size_in_bytes": 3000,
                        "expired": False,
                    }
                ]
            }
        if path == f"git/commits/{MERGE}":
            return {"parents": [{"sha": sha} for sha in self.parents]}
        if path == f"git/trees/{BASE}?recursive=1":
            return {
                "tree": (
                    [{"path": path, "type": "blob"} for path in CAPABILITIES]
                    if self.ready
                    else []
                )
            }
        if path.startswith("issues/3/comments?"):
            return copy.deepcopy(self.comments)
        raise AssertionError(endpoint)

    @staticmethod
    def assert_prefix(endpoint, prefix):
        if not endpoint.startswith(prefix):
            raise AssertionError(endpoint)

    def raw(self, endpoint, **kwargs):
        gate = endpoint.endswith("/2/zip")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "gate.json" if gate else "plan.json",
                json.dumps(self.report if gate else self.plan),
            )
        return buffer.getvalue()

    def write(self, endpoint, payload, **kwargs):
        self.writes.append((endpoint, payload, kwargs))
        if endpoint.endswith("/check-runs"):
            check = {
                **copy.deepcopy(payload),
                "id": max([50, *(item["id"] for item in self.checks)]) + 1,
                "app": {"id": APP_ID},
                "suite_attempt": self.run["run_attempt"],
            }
            self.checks.append(check)
        elif "/check-runs/" in endpoint:
            identity = int(endpoint.rsplit("/", 1)[1])
            check = next(item for item in self.checks if item["id"] == identity)
            check.update(copy.deepcopy(payload))
        else:
            return {"id": 50}
        job = next((item for item in self.jobs if item.get("id") == check["id"]), None)
        if job is None:
            job = {"id": check["id"]}
            self.jobs.append(job)
        job.update({key: check.get(key) for key in ("name", "status", "conclusion")})
        return copy.deepcopy(check)


class PRStatusTests(unittest.TestCase):
    def test_old_resolver_requests_upgrade_before_requesting_ci(self):
        for images_only in (False, True):
            with self.subTest(images_only=images_only):
                client = Client()
                client.main_sha = "e" * 40
                client.resolver_source = "def resolve(): pass\n"
                if images_only:
                    client.plan.update(
                        required_verification_ids=["quality"],
                        deferred_verifications=[],
                    )
                    client.refresh_report()
                result = reconcile(client, REPO, 101)
                self.assertEqual(result["state"], "upgrade")
                self.assertEqual(result["check"]["status"], "in_progress")
                self.assertNotIn("conclusion", result["check"])
                self.assertIn(MESSAGES["upgrade"], result["comment"])
                self.assertIn("Update the branch", result["check"]["output"]["title"])
                self.assertNotIn("can start the required checks", result["comment"])
                self.assertNotIn("main advancing alone", result["comment"])

    def test_old_resolver_updates_the_existing_guidance_comment(self):
        client = Client()
        client.resolver_source = "def resolve(): pass\n"
        client.comments = [
            {
                "id": 88,
                "body": MARKER + " Waiting for committer approval",
                "user": {"login": "github-actions[bot]", "type": "Bot"},
            }
        ]
        reconcile(client, REPO, 101)
        endpoint, payload, options = client.writes[-1]
        self.assertEqual(endpoint, f"repos/{REPO}/issues/comments/88")
        self.assertEqual(options, {"method": "PATCH"})
        self.assertIn(MESSAGES["upgrade"], payload["body"])

    def test_old_legacy_resolver_gets_upgrade_guidance_without_a_competing_gate(self):
        client = Client()
        client.ready = False
        client.resolver_source = "def resolve(): pass\n"
        client.run["conclusion"] = "failure"
        client.jobs = [
            {"name": name, "conclusion": "failure"} for name in ["CI / Gate", "PR Gate"]
        ]
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "legacy-upgrade")
        self.assertIsNone(result["check"])
        self.assertEqual(len(client.writes), 1)
        self.assertIn(MESSAGES["upgrade"], result["comment"])
        self.assertNotIn("can start the required checks", result["comment"])

    def test_resolver_upgrade_does_not_hide_a_genuine_test_failure(self):
        client = Client()
        client.resolver_source = "def resolve(): pass\n"
        client.run["conclusion"] = "failure"
        client.jobs[-1]["conclusion"] = "failure"
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "failure")
        self.assertEqual(result["check"]["conclusion"], "failure")
        self.assertIn(MESSAGES["upgrade"], result["comment"])
        self.assertNotIn("a committer must also comment", result["comment"])

    def test_completed_qualification_does_not_require_a_resolver_upgrade(self):
        client = Client()
        client.resolver_source = "def resolve(): pass\n"
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        with patch("pr_ci_status.require_current_resolver") as compatibility:
            result = reconcile(client, REPO, 101)
        compatibility.assert_not_called()
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["check"]["conclusion"], "success")

    def test_upgrade_dry_run_does_not_publish_changes(self):
        client = Client()
        client.resolver_source = "def resolve(): pass\n"
        result = reconcile(client, REPO, 101, dry_run=True)
        self.assertEqual(result["state"], "upgrade")
        self.assertIn(MESSAGES["upgrade"], result["comment"])
        self.assertEqual(client.writes, [])

    def test_fork_waiting_is_pending_with_actionable_comment(self):
        client = Client()
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "awaiting-approval")
        self.assertEqual(result["check"]["status"], "in_progress")
        self.assertNotIn("conclusion", result["check"])
        self.assertIn("/ci run", result["comment"])
        self.assertIn("e2e.dashboard", result["comment"])
        self.assertIn("Pending image builds: `router`, `dashboard`", result["comment"])
        self.assertIn("write, maintain or admin", result["comment"])
        self.assertEqual(len(client.writes), 2)

    def test_existing_bot_comment_updates_without_duplicate(self):
        client = Client()
        client.comments = [
            {
                "id": 88,
                "body": MARKER + " old",
                "user": {"login": "github-actions[bot]", "type": "Bot"},
            }
        ]
        reconcile(client, REPO, 101)
        self.assertEqual(client.writes[-1][0], f"repos/{REPO}/issues/comments/88")
        self.assertEqual(client.writes[-1][2], {"method": "PATCH"})

    def test_contributor_cannot_spoof_owned_comment(self):
        client = Client()
        client.comments = [
            {"id": 88, "body": MARKER, "user": {"login": "contributor", "type": "User"}}
        ]
        reconcile(client, REPO, 101)
        self.assertEqual(client.writes[-1][0], f"repos/{REPO}/issues/3/comments")

    def test_full_receipt_turns_gate_green_and_resolves_existing_comment(self):
        client = Client()
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        client.comments = [
            {
                "id": 88,
                "body": MARKER,
                "user": {"login": "github-actions[bot]", "type": "Bot"},
            }
        ]
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["check"]["conclusion"], "success")
        self.assertIn("All required CI qualification passed", result["comment"])

    def test_partial_approval_stays_pending(self):
        client = Client()
        client.plan["expected_verification_ids"].append("local.cli")
        client.plan["required_verification_ids"].append("local.cli")
        client.refresh_report()
        self.assertEqual(reconcile(client, REPO, 101)["state"], "awaiting-approval")

    def test_repeated_waiting_receipt_keeps_the_same_gate_pending(self):
        client = Client()
        for _ in range(2):
            result = reconcile(client, REPO, 101)
            self.assertEqual(result["state"], "awaiting-approval")
            self.assertEqual(result["check"]["status"], "in_progress")
        self.assertEqual(len(client.checks), 1)
        self.assertIsNone(client.jobs[-1]["conclusion"])

    def test_successful_rerun_ignores_its_pending_or_failed_published_gate(self):
        for conclusion in (None, "failure"):
            with self.subTest(conclusion=conclusion):
                client = Client()
                reconcile(client, REPO, 101)
                client.run.update(run_attempt=2, status="in_progress", conclusion=None)
                self.assertEqual(reconcile(client, REPO, 101)["state"], "running")
                client.jobs[-1]["conclusion"] = conclusion
                client.run.update(status="completed", conclusion="success")
                client.plan.update(
                    expected_verification_ids=["quality", "e2e.dashboard"],
                    deferred_verifications=[],
                    deferred_images=[],
                )
                client.refresh_report()
                for _ in range(2):
                    result = reconcile(client, REPO, 101)
                    self.assertEqual(result["check"]["conclusion"], "success")
                self.assertEqual(len(client.checks), 2)

    def test_unowned_failed_checks_are_still_ci_failures(self):
        for override in (
            {"app": {"id": APP_ID + 1}},
            {"external_id": f"{PREFIX}4:101:1"},
            {"external_id": None},
            {"name": "Another Gate"},
        ):
            with self.subTest(override=override):
                client = Client()
                check = {
                    "id": 77,
                    "name": "PR Gate",
                    "app": {"id": APP_ID},
                    "external_id": f"{PREFIX}3:101:1",
                    "status": "completed",
                    "conclusion": "failure",
                    **override,
                }
                client.checks = [check]
                client.jobs.append(
                    {key: check[key] for key in ("id", "name", "conclusion")}
                )
                self.assertEqual(reconcile(client, REPO, 101)["state"], "failure")

    def test_published_gate_does_not_replace_required_jobs_or_receipt_validation(self):
        for invalid in ("execution", "receipt"):
            with self.subTest(invalid=invalid):
                client = Client()
                reconcile(client, REPO, 101)
                if invalid == "execution":
                    client.jobs = [
                        job for job in client.jobs if job["name"] != "CI Execution"
                    ]
                else:
                    client.report["run_attempt"] = 0
                self.assertEqual(reconcile(client, REPO, 101)["state"], "failure")

    def test_real_failure_wins_over_deferred_approval(self):
        client = Client()
        client.run["conclusion"] = "failure"
        client.jobs[-1]["conclusion"] = "failure"
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["check"]["conclusion"], "failure")
        self.assertNotIn("Automatic checks passed", result["comment"])

    def test_images_only_waiting_is_visible_and_does_not_pass(self):
        client = Client()
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
        )
        client.refresh_report()
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "awaiting-approval")
        self.assertIn("Pending image builds", result["comment"])
        self.assertNotIn("conclusion", result["check"])

    def test_missing_or_stale_receipt_never_passes(self):
        for mutate in [
            lambda c: setattr(c, "missing_report", True),
            lambda c: c.report.update(run_attempt=0),
            lambda c: c.report.update(source_sha="f" * 40),
            lambda c: c.report.update(required_verification_ids=[]),
            lambda c: c.report.update(status="success"),
        ]:
            with self.subTest(mutate=mutate):
                client = Client()
                mutate(client)
                self.assertEqual(reconcile(client, REPO, 101)["state"], "failure")

    def test_wrong_merge_parents_and_markdown_inventory_are_rejected(self):
        client = Client()
        client.parents = [BASE, "f" * 40]
        self.assertEqual(reconcile(client, REPO, 101)["state"], "failure")
        client = Client()
        client.plan["deferred_verifications"] = ["@everyone\n/ci run"]
        client.refresh_report()
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "failure")
        self.assertNotIn("@everyone", json.dumps(result))

    def test_source_must_match_github_workflow_metadata_not_only_merge_parents(self):
        client = Client()
        client.plan["source_sha"] = "e" * 40
        client.refresh_report()
        self.assertEqual(reconcile(client, REPO, 101)["state"], "failure")
        client = Client()
        client.run["referenced_workflows"] = []
        self.assertEqual(reconcile(client, REPO, 101)["state"], "failure")

    def test_stale_head_or_newer_run_does_not_write(self):
        for mutate in [
            lambda c: c.pull["head"].update(sha="f" * 40),
            lambda c: setattr(c, "newer_run", True),
        ]:
            client = Client()
            mutate(client)
            self.assertIn("ignored", reconcile(client, REPO, 101))
            self.assertEqual(client.writes, [])

    def test_rewritten_main_stays_pending_and_requests_branch_update(self):
        client = Client()
        client.main_sha, client.base_ancestor = "e" * 40, "f" * 40
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "stale")
        self.assertEqual(result["check"]["status"], "in_progress")
        self.assertIn("Update the branch", result["comment"])

    def test_late_event_reads_current_attempt_before_publishing(self):
        client = Client()
        client.run.update(run_attempt=2, status="in_progress", conclusion=None)
        client.checks = [
            {
                "id": 77,
                "name": "PR Gate",
                "app": {"id": APP_ID},
                "external_id": f"{PREFIX}3:101:1",
                "status": "in_progress",
            }
        ]
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "running")
        self.assertTrue(result["check"]["external_id"].endswith(":2"))
        self.assertEqual(client.writes[0][0], f"repos/{REPO}/check-runs")
        self.assertEqual(client.writes[0][2], {})
        self.assertEqual(client.checks[0]["conclusion"], "neutral")
        self.assertEqual(client.checks[-1]["status"], "in_progress")

    def test_approval_completion_creates_a_new_gate_and_retires_old_pending_checks(
        self,
    ):
        client = Client()
        client.run["run_attempt"] = 3
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        client.checks = [
            {
                "id": identity,
                "name": "PR Gate",
                "app": {"id": APP_ID},
                "external_id": f"{PREFIX}{external}",
                "status": status,
            }
            for identity, external, status in (
                (77, "3:101:1", "in_progress"),
                (78, "3:101:2", "in_progress"),
                (79, "3:101:2", "completed"),
                (80, "3:102:2", "in_progress"),
                (81, "4:101:2", "in_progress"),
            )
        ]
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "success")
        endpoint, payload, options = client.writes[0]
        self.assertEqual(endpoint, f"repos/{REPO}/check-runs")
        self.assertEqual(options, {})
        self.assertEqual(payload["external_id"], f"{PREFIX}3:101:3")
        self.assertEqual(payload["conclusion"], "success")
        self.assertEqual(
            [item[0] for item in client.writes if "/check-runs" in item[0]],
            [
                f"repos/{REPO}/check-runs",
                f"repos/{REPO}/check-runs/77",
                f"repos/{REPO}/check-runs/78",
            ],
        )
        for _, retirement, _ in client.writes[1:3]:
            self.assertEqual(
                retirement, {"status": "completed", "conclusion": "neutral"}
            )
        self.assertEqual(client.checks[-1]["external_id"], f"{PREFIX}3:101:3")
        self.assertEqual(client.checks[-1]["conclusion"], "success")
        self.assertEqual(client.checks[0]["external_id"], f"{PREFIX}3:101:1")
        self.assertEqual(client.checks[1]["external_id"], f"{PREFIX}3:101:2")

    def test_rerun_verdict_is_visible_to_branch_protection(self):
        for state in ("success", "failure", "awaiting-approval"):
            with self.subTest(state=state):
                client = Client()
                reconcile(client, REPO, 101)
                old_id = client.checks[0]["id"]
                client.run.update(
                    run_attempt=2,
                    conclusion="failure" if state == "failure" else "success",
                )
                if state != "awaiting-approval":
                    client.plan.update(
                        expected_verification_ids=["quality", "e2e.dashboard"],
                        deferred_verifications=[],
                        deferred_images=[],
                    )
                client.refresh_report()
                for _ in range(2):
                    self.assertEqual(reconcile(client, REPO, 101)["state"], state)
                    visible = client.request(
                        f"repos/{REPO}/commits/{HEAD}/check-runs?filter=latest"
                    )["check_runs"]
                    self.assertEqual(len(visible), 1)
                    self.assertNotEqual(visible[0]["id"], old_id)
                    if state == "awaiting-approval":
                        self.assertEqual(visible[0]["status"], "in_progress")
                        self.assertNotIn("conclusion", visible[0])
                    else:
                        self.assertEqual(visible[0]["conclusion"], state)
                self.assertEqual(len(client.checks), 2)
                self.assertEqual(client.checks[0]["conclusion"], "neutral")

    def test_approved_rerun_after_main_advances_publishes_the_tested_revision(self):
        client = Client()
        reconcile(client, REPO, 101)
        client.main_sha = "e" * 40
        client.pull["base"]["sha"] = client.main_sha
        client.pull["merge_commit_sha"] = "f" * 40
        client.run["run_attempt"] = 2
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["check"]["head_sha"], HEAD)
        visible = client.request(
            f"repos/{REPO}/commits/{HEAD}/check-runs?filter=latest"
        )["check_runs"]
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["conclusion"], "success")
        summary = result["check"]["output"]["summary"]
        self.assertIn(f"Tested merge `{MERGE[:12]}`", summary)
        self.assertIn(f"against base `{BASE[:12]}`", summary)
        self.assertNotIn(client.main_sha[:12], summary)

    def test_rewritten_main_before_publication_cannot_publish_success(self):
        client = Client()
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        with patch("pr_ci_status.main_contains", side_effect=[True, False]):
            self.assertIn("ignored", reconcile(client, REPO, 101))
        self.assertEqual(client.writes, [])

    def test_new_contributor_head_before_publication_cannot_publish_success(self):
        client = Client()
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        original = client.request
        reads = 0
        revalidation_read = 2

        def request(endpoint, **kwargs):
            nonlocal reads
            result = original(endpoint, **kwargs)
            if endpoint.endswith("/pulls/3"):
                reads += 1
                if reads == revalidation_read:
                    result["head"]["sha"] = "f" * 40
            return result

        client.request = request
        self.assertIn("ignored", reconcile(client, REPO, 101))
        self.assertEqual(client.writes, [])

    def test_reconciliation_replaces_a_success_hidden_by_an_earlier_attempt(self):
        client = Client()
        reconcile(client, REPO, 101)
        # Model the broken publisher: attempt 2 patched attempt 1's check.
        client.checks[0].update(
            external_id=f"{PREFIX}3:101:2", status="completed", conclusion="success"
        )
        client.run["run_attempt"] = 2
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        for _ in range(2):
            reconcile(client, REPO, 101)
        visible = client.request(
            f"repos/{REPO}/commits/{HEAD}/check-runs?filter=latest"
        )["check_runs"]
        self.assertEqual(len(client.checks), 2)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible[0]["conclusion"], "success")
        self.assertEqual(visible[0]["external_id"], f"{PREFIX}3:101:2")

    def test_completed_or_different_run_or_pr_check_is_not_reused(self):
        for external, status in (
            ("3:101:1", "completed"),
            ("3:102:1", "in_progress"),
            ("3:1010:1", "in_progress"),
            ("4:101:1", "in_progress"),
        ):
            with self.subTest(external=external, status=status):
                client = Client()
                client.run["run_attempt"] = 2
                client.refresh_report()
                client.checks = [
                    {
                        "id": 77,
                        "name": "PR Gate",
                        "app": {"id": APP_ID},
                        "external_id": f"{PREFIX}{external}",
                        "status": status,
                    }
                ]
                reconcile(client, REPO, 101)
                endpoint, payload, options = client.writes[0]
                self.assertEqual(endpoint, f"repos/{REPO}/check-runs")
                self.assertEqual(options, {})
                self.assertEqual(payload["external_id"], f"{PREFIX}3:101:2")
                self.assertEqual(payload["head_sha"], HEAD)

    def test_exact_current_check_takes_priority_over_newer_pending_check(self):
        client = Client()
        client.run["run_attempt"] = 2
        client.refresh_report()
        client.checks = [
            {
                "id": identity,
                "name": "PR Gate",
                "app": {"id": APP_ID},
                "external_id": f"{PREFIX}{external}",
                "status": "in_progress",
            }
            for identity, external in ((77, "3:101:2"), (78, "3:101:1"))
        ]
        reconcile(client, REPO, 101)
        self.assertEqual(client.writes[0][0], f"repos/{REPO}/check-runs/77")
        self.assertEqual(client.writes[0][2], {"method": "PATCH"})
        self.assertEqual(client.checks[0]["external_id"], f"{PREFIX}3:101:2")
        self.assertEqual(client.checks[1]["conclusion"], "neutral")

    def test_initial_attempt_does_not_invent_plan_or_compete_with_legacy_gate(self):
        client = Client()
        client.run.update(status="in_progress", conclusion=None)
        self.assertIn("ignored", reconcile(client, REPO, 101))
        self.assertEqual(client.writes, [])

    def test_legacy_run_gets_guidance_but_no_competing_required_check(self):
        client = Client()
        client.ready = False
        client.run["conclusion"] = "failure"
        client.jobs = [
            {"name": name, "conclusion": "failure"} for name in ["CI / Gate", "PR Gate"]
        ]
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "legacy")
        self.assertIsNone(result["check"])
        self.assertEqual(len(client.writes), 1)
        self.assertIn("Update the branch", result["comment"])

    def test_dry_run_never_mutates(self):
        client = Client()
        self.assertEqual(
            reconcile(client, REPO, 101, dry_run=True)["state"], "awaiting-approval"
        )
        self.assertEqual(client.writes, [])

    def test_superseded_during_collection_does_not_write(self):
        client = Client()
        with patch("pr_ci_status.latest_run", side_effect=[True, False]):
            self.assertIn("ignored", reconcile(client, REPO, 101))
        self.assertEqual(client.writes, [])

    def test_base_or_merge_metadata_changing_does_not_discard_a_valid_plan(self):
        revalidation_read = 2
        for field in ("base", "merge"):
            with self.subTest(field=field):
                client = Client()
                original = client.request
                reads = 0

                def request(endpoint, *, original=original, field=field, **kwargs):
                    nonlocal reads
                    result = original(endpoint, **kwargs)
                    if endpoint.endswith("/pulls/3"):
                        reads += 1
                        if reads == revalidation_read:
                            if field == "base":
                                result["base"]["sha"] = "f" * 40
                            else:
                                result["merge_commit_sha"] = "f" * 40
                    return result

                client.request = request
                self.assertEqual(
                    reconcile(client, REPO, 101)["state"], "awaiting-approval"
                )
                self.assertTrue(client.writes)

    def test_manual_dispatch_reads_run_id_from_event_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            event.write_text(json.dumps({"inputs": {"run_id": "101"}}))
            with (
                patch("sys.argv", ["pr_ci_status.py", "--event", str(event)]),
                patch.dict("os.environ", {"GITHUB_REPOSITORY": REPO}),
                patch("pr_ci_status.reconcile", return_value={}) as reconcile_mock,
                patch("builtins.print"),
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(reconcile_mock.call_args.args[1:], (REPO, 101))


if __name__ == "__main__":
    unittest.main()
