from __future__ import annotations

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
from manual_ci import ManualCIError
from pr_ci_status import (
    APP_ID,
    CAPABILITIES,
    MARKER,
    PREFIX,
    diagnose,
    main,
    reconcile,
    required_checks,
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
        self.compare_error = None
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
            if self.compare_error is not None:
                raise self.compare_error
            return {"merge_base_commit": {"sha": self.base_ancestor}}
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
                checks = [
                    item
                    for item in checks
                    if item.get("run_attempt", self.run["run_attempt"])
                    == self.run["run_attempt"]
                ]
            return {"check_runs": copy.deepcopy(checks)}
        if path.startswith("actions/runs/101/attempts/"):
            return {
                "jobs": [
                    copy.deepcopy(item)
                    for item in self.jobs
                    if item.get("run_attempt", self.run["run_attempt"])
                    == self.run["run_attempt"]
                ]
            }
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
                # GitHub's attempt membership survives PATCH, including an
                # external_id rewrite, and controls filter=latest visibility.
                "run_attempt": self.run["run_attempt"],
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
        job["run_attempt"] = check.get("run_attempt", self.run["run_attempt"])
        return copy.deepcopy(check)


class PRStatusTests(unittest.TestCase):
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

    def test_rerun_replaces_hidden_check_even_with_current_external_id(self):
        endpoint = f"repos/{REPO}/commits/{HEAD}/check-runs"
        for old_status in ("in_progress", "completed"):
            for verdict in ("success", "failure"):
                with self.subTest(old_status=old_status, verdict=verdict):
                    client = Client()
                    reconcile(client, REPO, 101)
                    old = client.checks[0]
                    # Reproduce an already affected PR: the old publisher
                    # changed the external_id but not the check's attempt.
                    old.update(external_id=f"{PREFIX}3:101:2", status=old_status)
                    if old_status == "completed":
                        old["conclusion"] = "success"
                    client.run.update(run_attempt=2, conclusion=verdict)
                    client.plan.update(
                        expected_verification_ids=["quality", "e2e.dashboard"],
                        deferred_verifications=[],
                        deferred_images=[],
                    )
                    client.refresh_report()
                    self.assertEqual(
                        client.request(f"{endpoint}?filter=latest")["check_runs"], []
                    )
                    for _ in range(2):
                        result = reconcile(client, REPO, 101)
                        self.assertEqual(result["state"], verdict)
                        latest = client.request(f"{endpoint}?filter=latest")[
                            "check_runs"
                        ]
                        self.assertEqual(len(latest), 1)
                        self.assertNotEqual(latest[0]["id"], old["id"])
                        self.assertEqual(latest[0]["conclusion"], verdict)
                        self.assertEqual(latest[0]["run_attempt"], 2)
                    self.assertEqual(len(client.checks), 2)
                    self.assertEqual(old["run_attempt"], 1)

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

    def test_stale_head_or_newer_run_does_not_write(self):
        for mutate in [
            lambda c: c.pull["head"].update(sha="f" * 40),
            lambda c: setattr(c, "newer_run", True),
        ]:
            client = Client()
            mutate(client)
            self.assertIn("ignored", reconcile(client, REPO, 101))
            self.assertEqual(client.writes, [])

    def test_changed_base_stays_pending_and_requests_branch_update(self):
        client = Client()
        client.main_sha = "f" * 40
        client.base_ancestor = "e" * 40
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "stale")
        self.assertEqual(result["check"]["status"], "in_progress")
        self.assertIn("Update the branch", result["comment"])

    def test_unreachable_base_compare_api_error_marks_gate_stale(self):
        client = Client()
        client.main_sha = "f" * 40
        client.compare_error = ManualCIError(
            "Comparison not found (HTTP 404).", http_status=404
        )
        client.pull["merge_commit_sha"] = "d" * 40
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "stale")
        self.assertEqual(result["check"]["status"], "in_progress")
        self.assertIn("Update the branch", result["comment"])

    def test_advanced_main_with_unchanged_head_does_not_mark_gate_stale(self):
        client = Client()
        client.main_sha = "f" * 40
        client.base_ancestor = BASE
        client.pull["merge_commit_sha"] = "d" * 40
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "awaiting-approval")
        self.assertEqual(result["check"]["status"], "in_progress")
        self.assertNotIn("Update the branch", result["comment"])

    def test_advanced_main_with_passed_verdict_publishes_success(self):
        client = Client()
        client.main_sha = "f" * 40
        client.base_ancestor = BASE
        client.pull["merge_commit_sha"] = "d" * 40
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
        result = reconcile(client, REPO, 101)
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["check"]["conclusion"], "success")

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
        self.assertEqual(client.checks[0]["external_id"], f"{PREFIX}3:101:1")
        self.assertEqual(client.checks[0]["conclusion"], "neutral")
        self.assertEqual(client.checks[1]["status"], "in_progress")
        self.assertNotIn("conclusion", client.checks[1])

    def test_approval_completion_retires_only_same_run_pending_checks(self):
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
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["conclusion"], "success")
        self.assertEqual(payload["head_sha"], HEAD)
        retired = client.writes[1:]
        self.assertEqual(
            [endpoint for endpoint, _, _ in retired],
            [f"repos/{REPO}/check-runs/{identity}" for identity in (77, 78)],
        )
        for _, update, options in retired:
            self.assertEqual(options, {"method": "PATCH"})
            self.assertEqual(update["status"], "completed")
            self.assertEqual(update["conclusion"], "neutral")
            self.assertNotIn("head_sha", update)
            self.assertNotIn("external_id", update)

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

    def test_base_or_merge_changing_before_write_cannot_publish_success(self):
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
                self.assertIn("ignored", reconcile(client, REPO, 101))
                self.assertEqual(client.writes, [])

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


class DiagnosticClient(Client):
    """Explicit read-only same-head checks, suite runs and exact-attempt jobs."""

    def __init__(self):
        super().__init__()
        self.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        self.refresh_report()
        self.run["repository"] = {"full_name": REPO}
        self.run["check_suite_id"] = 20
        self.native_run = {
            **copy.deepcopy(self.run),
            "id": 99,
            "check_suite_id": 19,
            "conclusion": "cancelled",
        }
        self.native_jobs = [
            {"id": 70, "name": "PR Gate", "conclusion": "failure"},
            {"id": 71, "name": "CI / Gate", "conclusion": "failure"},
            {"id": 72, "name": "Quality", "conclusion": "cancelled"},
        ]
        self.checks = [
            self.check(80, f"{PREFIX}3:101:1", "success", 19),
            self.check(70, "12345678-1234-1234-1234-123456789abc", "failure", 19),
        ]
        self.visible = {70, 80}
        self.required = {70: True, 80: True}
        self.required_complete = True
        self.reads = []
        self.mutate_on_final = None
        self.run_reads = {}
        self.live_base = BASE

    def advance_main_unrelated(self):
        self.live_base = "d" * 40
        self.base_ancestor = "f" * 40

    @staticmethod
    def check(identity, external, conclusion, suite):
        return {
            "id": identity,
            "name": "PR Gate",
            "head_sha": HEAD,
            "app": {"id": APP_ID},
            "external_id": external,
            "check_suite": {"id": suite},
            "status": "completed",
            "conclusion": conclusion,
        }

    def required_checks(self, repo, head, number):
        assert (repo, head, number) == (REPO, HEAD, 3)
        return copy.deepcopy(self.required), self.required_complete

    def request(self, endpoint, **kwargs):
        assert kwargs.get("method", "GET") == "GET"
        self.reads.append(endpoint)
        path = endpoint.removeprefix(f"repos/{REPO}/")
        if path.startswith(f"commits/{HEAD}/check-runs?"):
            checks = self.checks
            if "filter=latest" in path:
                checks = [c for c in checks if c["id"] in self.visible]
            return {"check_runs": copy.deepcopy(checks)}
        if path.startswith("actions/runs?check_suite_id="):
            return {"workflow_runs": [copy.deepcopy(self.native_run)]}
        if path == "actions/runs/99":
            return copy.deepcopy(self.native_run)
        if path.startswith("actions/runs/99/attempts/"):
            jobs = self.native_jobs if "/attempts/1/" in path else []
            return {"jobs": copy.deepcopy(jobs)}
        if path == "git/ref/heads/main":
            return {
                "ref": "refs/heads/main",
                "object": {"type": "commit", "sha": self.live_base},
            }
        if path == "actions/runs/101":
            self.run_reads[path] = self.run_reads.get(path, 0) + 1
            if self.run_reads[path] > 1 and self.mutate_on_final:
                self.mutate_on_final(self)
        return super().request(endpoint, **kwargs)

    def write(self, *args, **kwargs):
        raise AssertionError("Diagnostic attempted a write")


class DiagnosticTests(unittest.TestCase):
    def diagnose(self, client):
        with (
            patch("pr_ci_status.reconcile", side_effect=AssertionError("reconcile")),
            patch("pr_ci_status.publish", side_effect=AssertionError("publish")),
            patch("subprocess.run", side_effect=AssertionError("unmocked subprocess")),
        ):
            return diagnose(client, REPO, 3, HEAD, 101, 1)

    def test_confirmed_competition_does_not_prove_cancellation_cause(self):
        result = self.diagnose(DiagnosticClient())
        self.assertEqual(result["state"], "competing_required")
        self.assertTrue(result["evidence"]["native_failures_verified"])
        self.assertIsNone(result["evidence"]["cancellation_only"])
        self.assertIn("cancellation_cause_unproven", result["reason_codes"])
        self.assertTrue(result["evidence"]["full_qualification"])
        self.assertFalse(result["actionable"])
        self.assertEqual(result["checks"]["all"][1]["role"], "native")
        self.assertEqual(result["checks"]["all"][0]["suite_id"], 19)

    def test_healthy_without_native_failure_does_not_claim_recovery(self):
        client = DiagnosticClient()
        client.checks.pop()
        result = self.diagnose(client)
        self.assertEqual(result["state"], "healthy")
        self.assertFalse(result["evidence"]["competition"])
        self.assertFalse(result["actionable"])

    def test_all_exposes_hidden_native_and_publisher_checks(self):
        for visible in ({80}, {70}, {70, 80}):
            with self.subTest(visible=visible):
                client = DiagnosticClient()
                client.visible = visible
                result = self.diagnose(client)
                self.assertEqual(result["state"], "competing_required")
                self.assertEqual(len(result["checks"]["all"]), 2)
                self.assertEqual(len(result["checks"]["latest"]), len(visible))

    def test_running_qualification_is_not_reported_as_failed(self):
        for status in ("queued", "in_progress"):
            with self.subTest(status=status):
                client = DiagnosticClient()
                client.run.update(status=status, conclusion=None)
                result = self.diagnose(client)
                self.assertEqual(result["state"], "running")
                self.assertIn("qualification_incomplete", result["reason_codes"])
                self.assertFalse(result["evidence"]["full_qualification"])
                self.assertTrue(result["evidence"]["snapshot_current"])

    def test_running_publisher_check_is_not_reported_as_failed(self):
        client = DiagnosticClient()
        client.run.update(status="in_progress", conclusion=None)
        client.checks[0].update(status="in_progress", conclusion=None)
        result = self.diagnose(client)
        self.assertEqual(result["state"], "running")
        self.assertIn("qualification_incomplete", result["reason_codes"])

    def test_hidden_awaiting_approval_gate_stays_deferred(self):
        client = DiagnosticClient()
        client.plan.update(
            expected_verification_ids=["quality"],
            required_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=["e2e.dashboard"],
        )
        client.refresh_report()
        client.checks[0].update(status="in_progress", conclusion=None)
        client.visible = {70}
        result = self.diagnose(client)
        self.assertEqual(result["state"], "deferred")
        self.assertIn("qualification_deferred", result["reason_codes"])
        self.assertFalse(result["evidence"]["full_qualification"])
        self.assertTrue(result["evidence"]["snapshot_current"])
        self.assertEqual(len(result["checks"]["all"]), 2)
        self.assertEqual(len(result["checks"]["latest"]), 1)

    def test_failed_current_qualification_stays_failed(self):
        for field in ("run", "native_job", "own_check"):
            with self.subTest(field=field):
                client = DiagnosticClient()
                if field == "run":
                    client.run["conclusion"] = "failure"
                elif field == "native_job":
                    client.jobs.append(client.native_jobs[0])
                else:
                    client.checks[0]["conclusion"] = "failure"
                self.assertEqual(self.diagnose(client)["state"], "failed")

    def test_wrong_app_or_unproved_native_is_uncertain(self):
        for mutation in (
            lambda c: c.checks[1]["app"].update(id=1),
            lambda c: c.checks[1].update(external_id="private raw payload"),
            lambda c: c.native_jobs[0].update(id=700),
            lambda c: c.native_run.update(path=".github/workflows/other.yml"),
            lambda c: c.native_run.update(head_branch="other"),
            lambda c: c.native_run.update(run_attempt=2),
        ):
            with self.subTest(mutation=mutation):
                client = DiagnosticClient()
                mutation(client)
                result = self.diagnose(client)
                self.assertEqual(result["state"], "uncertain")
                self.assertFalse(result["evidence"]["cancellation_only"])
                self.assertNotIn("private raw payload", json.dumps(result))

    def test_closed_pull_is_reported_as_not_open(self):
        client = DiagnosticClient()
        client.pull.update(state="closed")
        result = self.diagnose(client)
        self.assertEqual(result["state"], "ignored")
        self.assertEqual(result["reason_codes"], ["pull_not_open"])

    def test_explicit_identity_mismatches_stop(self):
        for mutation in (
            lambda c: c.pull.update(number=4),
            lambda c: c.run.update(head_sha="d" * 40),
            lambda c: c.run.update(workflow_id=9),
            lambda c: c.run.update(path=".github/workflows/other.yml"),
            lambda c: c.run.update(run_attempt=2),
            lambda c: c.run["repository"].update(full_name="other/router"),
            lambda c: setattr(c, "newer_run", True),
        ):
            with self.subTest(mutation=mutation):
                client = DiagnosticClient()
                mutation(client)
                self.assertEqual(self.diagnose(client)["state"], "ignored")

    def test_deferred_or_unvalidated_receipts_never_qualify(self):
        for mutation, state in (
            (lambda c: c.plan.update(deferred_images=["router"]), "deferred"),
            (lambda c: setattr(c, "missing_report", True), "evidence_unavailable"),
            (lambda c: c.report.update(run_attempt=9), "evidence_unavailable"),
            (lambda c: c.parents.reverse(), "evidence_unavailable"),
        ):
            with self.subTest(state=state):
                client = DiagnosticClient()
                mutation(client)
                if state == "deferred":
                    client.refresh_report()
                result = self.diagnose(client)
                self.assertEqual(result["state"], state)
                self.assertFalse(result["evidence"]["full_qualification"])

    def test_tampered_plan_is_not_echoed(self):
        client = DiagnosticClient()
        client.plan["source_sha"] = "secret-private-host"
        result = self.diagnose(client)
        self.assertEqual(result["state"], "evidence_unavailable")
        self.assertNotIn("secret-private-host", json.dumps(result))

    def test_current_source_uses_shared_plan_selection_contract(self):
        client = DiagnosticClient()
        live_identity = {
            "head_sha": HEAD,
            "base_sha": "d" * 40,
            "source_sha": "e" * 40,
        }
        with (
            patch("pr_ci_status.source_identity", return_value=live_identity),
            patch("pr_ci_status.plan_for_run", return_value=client.plan) as select_plan,
        ):
            result = self.diagnose(client)
        self.assertEqual(result["state"], "competing_required")
        self.assertTrue(result["evidence"]["current_source"])
        self.assertTrue(result["evidence"]["snapshot_current"])
        self.assertEqual(select_plan.call_count, 2)

    def test_current_source_change_during_collection_invalidates_snapshot(self):
        client = DiagnosticClient()
        live_identity = {
            "head_sha": HEAD,
            "base_sha": "d" * 40,
            "source_sha": "e" * 40,
        }
        with (
            patch("pr_ci_status.source_identity", return_value=live_identity),
            patch("pr_ci_status.plan_for_run", side_effect=[client.plan, None]),
        ):
            result = self.diagnose(client)
        self.assertEqual(result["state"], "stale")
        self.assertIn("source_changed", result["reason_codes"])
        self.assertFalse(result["evidence"]["snapshot_current"])

    def test_ancestor_base_preserves_current_competition(self):
        client = DiagnosticClient()
        client.live_base = "d" * 40
        result = self.diagnose(client)
        self.assertEqual(result["state"], "competing_required")
        self.assertTrue(result["evidence"]["competition"])
        self.assertTrue(result["evidence"]["current_source"])
        self.assertTrue(result["evidence"]["snapshot_current"])
        self.assertNotIn("source_stale", result["reason_codes"])

    def test_unrelated_base_preserves_raw_competition_but_no_readiness(self):
        client = DiagnosticClient()
        client.advance_main_unrelated()
        result = self.diagnose(client)
        self.assertEqual(result["state"], "stale_blockers")
        self.assertTrue(result["evidence"]["competition"])
        self.assertFalse(result["evidence"]["current_source"])
        self.assertIn("source_stale", result["reason_codes"])

    def test_requiredness_change_during_collection_invalidates_snapshot(self):
        client = DiagnosticClient()
        reads = 0

        def required_checks(repo, head, number):
            nonlocal reads
            self.assertEqual((repo, head, number), (REPO, HEAD, 3))
            reads += 1
            if reads == 1:
                return {70: True, 80: True}, True
            return {70: False, 80: False}, True

        client.required_checks = required_checks
        result = self.diagnose(client)
        self.assertEqual(result["state"], "stale")
        self.assertIn("requiredness_changed", result["reason_codes"])
        self.assertFalse(result["evidence"]["snapshot_current"])
        self.assertEqual(reads, 2)

    def test_stable_requiredness_is_revalidated(self):
        client = DiagnosticClient()
        reads = 0
        original = client.required_checks

        def required_checks(repo, head, number):
            nonlocal reads
            reads += 1
            return original(repo, head, number)

        client.required_checks = required_checks
        result = self.diagnose(client)
        self.assertEqual(result["state"], "competing_required")
        self.assertTrue(result["evidence"]["snapshot_current"])
        self.assertEqual(reads, 2)

    def test_requiredness_unavailable_or_incomplete_is_not_inferred(self):
        for unavailable in (False, True):
            client = DiagnosticClient()
            client.required_complete = False
            if unavailable:
                client.required_checks = lambda *a: (_ for _ in ()).throw(
                    RuntimeError("private API payload")
                )
            result = self.diagnose(client)
            self.assertEqual(result["state"], "suspected_competition")
            self.assertFalse(result["evidence"]["requiredness_complete"])
            self.assertIsNone(result["checks"]["all"][0]["required"])
            self.assertNotIn("private API payload", json.dumps(result))

    def test_reconciliation_failure_followed_by_cancel_is_not_only_cancel(self):
        client = DiagnosticClient()
        client.native_jobs[0]["steps"] = [
            {"name": "Require successful reconciliation", "conclusion": "failure"}
        ]
        result = self.diagnose(client)
        self.assertTrue(result["evidence"]["native_failures_verified"])
        self.assertIsNone(result["evidence"]["cancellation_only"])
        self.assertIn("cancellation_cause_unproven", result["reason_codes"])
        self.assertFalse(result["actionable"])

    def test_mixed_real_failure_is_not_cancellation_only(self):
        client = DiagnosticClient()
        client.native_jobs.append(
            {"id": 73, "name": "Unit tests", "conclusion": "failure"}
        )
        result = self.diagnose(client)
        self.assertEqual(result["state"], "uncertain")
        self.assertFalse(result["evidence"]["cancellation_only"])

    def test_final_snapshot_change_stops_classification(self):
        for mutate in (
            lambda c: c.pull["head"].update(sha="d" * 40),
            lambda c: c.pull["base"].update(sha="d" * 40),
            lambda c: c.pull.update(merge_commit_sha="d" * 40),
            lambda c: c.run.update(run_attempt=2),
            lambda c: c.run.update(status="in_progress"),
            lambda c: c.native_run.update(run_attempt=2),
            lambda c: setattr(c, "newer_run", True),
            lambda c: c.advance_main_unrelated(),
            lambda c: c.checks[0].update(conclusion="failure"),
        ):
            with self.subTest(mutate=mutate):
                client = DiagnosticClient()
                client.mutate_on_final = mutate
                result = self.diagnose(client)
                self.assertEqual(result["state"], "stale")
                self.assertFalse(result["actionable"])
                self.assertFalse(result["evidence"]["snapshot_current"])

    def test_failed_run_missing_artifact_still_reports_failure(self):
        client = DiagnosticClient()
        client.run["conclusion"] = "failure"
        client.missing_report = True
        self.assertEqual(self.diagnose(client)["state"], "failed")

    def test_fixed_requiredness_query_batches_by_node_identity_read_only(self):
        client = DiagnosticClient()
        checks = [
            {"id": identity, "node_id": f"CR_{identity}"} for identity in range(1, 103)
        ]
        client.request = lambda endpoint: {
            "check_runs": checks[:100] if endpoint.endswith("&page=1") else checks[100:]
        }
        calls = []

        def gh(arguments, *, read, text):
            self.assertTrue(read)
            self.assertTrue(text)
            self.assertNotIn("mutation", arguments)
            calls.append(arguments)
            ids = [
                int(arg.rsplit("_", 1)[1])
                for arg in arguments
                if arg.startswith("ids[]=")
            ]
            return type(
                "Response",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "data": {
                                "nodes": [
                                    {
                                        "databaseId": identity,
                                        "isRequired": identity % 2 == 0,
                                    }
                                    for identity in ids
                                ]
                            }
                        }
                    ),
                },
            )()

        client.gh = gh
        mapping, complete = required_checks(client, REPO, HEAD, 3)
        self.assertTrue(complete)
        self.assertEqual(len(mapping), 102)
        self.assertEqual(len(calls), 2)
        self.assertFalse(mapping[1])
        self.assertTrue(mapping[102])

    def test_requiredness_partial_graphql_error_discards_all_requiredness(self):
        client = DiagnosticClient()
        client.request = lambda endpoint: {
            "check_runs": [{"id": 80, "node_id": "CR_80"}]
        }
        client.gh = lambda arguments, **kwargs: type(
            "Response",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(
                    {"errors": [{"message": "private"}], "data": {"nodes": []}}
                ),
            },
        )()
        self.assertEqual(required_checks(client, REPO, HEAD, 3), ({}, False))

    def test_diagnostic_flags_cannot_fall_through_to_publication(self):
        for arguments in (
            ["--pr", "3", "--run-id", "101"],
            ["--head", HEAD, "--run-id", "101"],
            ["--attempt", "1", "--run-id", "101"],
            ["--diagnose", "--pr", "3", "--run-id", "101"],
            ["--diagnose", "--event", "event.json"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("sys.argv", ["pr_ci_status.py", *arguments]),
                patch(
                    "pr_ci_status.StatusClient", side_effect=AssertionError("client")
                ),
                patch("pr_ci_status.reconcile", side_effect=AssertionError("publish")),
                patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 2)

    def test_diagnose_cli_is_read_only_even_without_dry_run(self):
        client = DiagnosticClient()
        output = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "pr_ci_status.py",
                    "--diagnose",
                    "--pr",
                    "3",
                    "--head",
                    HEAD,
                    "--run-id",
                    "101",
                    "--attempt",
                    "1",
                ],
            ),
            patch.dict("os.environ", {"GITHUB_REPOSITORY": REPO}),
            patch("pr_ci_status.StatusClient", return_value=client),
            patch("sys.stdout", output),
            patch("subprocess.run", side_effect=AssertionError("subprocess")),
        ):
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue())["state"], "competing_required")


if __name__ == "__main__":
    unittest.main()
