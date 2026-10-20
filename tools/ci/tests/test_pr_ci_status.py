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
            return {"check_runs": copy.deepcopy(self.checks)}
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
                self.assertEqual(len(client.checks), 1)

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
        client.pull["merge_commit_sha"] = "f" * 40
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
        self.assertEqual(client.writes[0][0], f"repos/{REPO}/check-runs/77")
        self.assertEqual(client.writes[0][2], {"method": "PATCH"})

    def test_approval_completion_resolves_latest_pending_check_in_the_same_run(self):
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
        self.assertEqual(endpoint, f"repos/{REPO}/check-runs/78")
        self.assertEqual(options, {"method": "PATCH"})
        self.assertEqual(payload["external_id"], f"{PREFIX}3:101:3")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["conclusion"], "success")
        self.assertNotIn("head_sha", payload)

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


if __name__ == "__main__":
    unittest.main()
