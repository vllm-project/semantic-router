"""The privileged status publisher must never execute a PR's contents."""

from __future__ import annotations

import concurrent.futures
import copy
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))

import workflow_policy_validation  # noqa: E402
from pr_ci_events import resolve_event  # noqa: E402
from pr_ci_pending import recover_pending  # noqa: E402
from pr_ci_pending_fixtures import PendingClient  # noqa: E402
from pr_ci_status import reconcile  # noqa: E402
from test_pr_ci_pending import invoke  # noqa: E402
from test_pr_ci_status import REPO  # noqa: E402
from validate_workflows import load_workflows  # noqa: E402
from workflow_policy_validation import (  # noqa: E402
    validate_gate_transport,
    validate_mergify_contract,
    validate_pr_contract,
    validate_publisher_bootstrap,
    validate_status_publisher,
)


class CIStatusWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        errors: list[str] = []
        self.workflows = load_workflows(errors)
        self.assertEqual(errors, [])
        self.publisher = self.workflows["ci-status.yml"]

    def lock_key(self, job_name, run_id):
        identity_expression = {
            "publish-pr": "${{ github.event.workflow_run.id }}",
            "publish": "${{ needs.resolve.outputs.run_id }}",
        }
        return self.publisher.jobs[job_name]["concurrency"]["group"].replace(
            identity_expression[job_name], str(run_id)
        )

    def test_pull_request_publication_has_no_separate_resolver_job(self):
        job = self.publisher.jobs.get("publish-pr")
        self.assertIsNotNone(
            job, "Pull Request observations need a single-job publisher"
        )
        self.assertNotIn("needs", job)
        self.assertEqual(
            job["concurrency"]["group"], "ci-status-${{ github.event.workflow_run.id }}"
        )
        self.assertIn(
            "github.event.workflow_run.path == '.github/workflows/community.yml'",
            self.publisher.jobs["resolve"]["if"],
        )

    def test_related_events_serialize_on_the_qualification_run(self):
        client = PendingClient()
        sibling = invoke(client, resolve=True)
        normal = {"action": "completed", "workflow_run": copy.deepcopy(client.run)}
        resolved = invoke(client, normal, resolve=True)
        self.assertEqual(resolved, {"run_id": 101, "mode": "publish"})
        keys = [
            self.lock_key("publish", sibling["run_id"]),
            self.lock_key("publish-pr", normal["workflow_run"]["id"]),
        ]
        self.assertEqual(keys, ["ci-status-101", "ci-status-101"])

    def test_resolver_has_no_write_permission(self):
        job = self.publisher.jobs.get("resolve")
        self.assertIsNotNone(job)
        self.assertEqual(set(job["permissions"].values()), {"read"})

    def test_commit_status_rerun_shares_the_publisher_lock_for_the_authorized_run(self):
        rerun = self.workflows["manual-ci.yml"].jobs["rerun"]
        self.assertEqual(
            rerun["concurrency"]["group"],
            "ci-status-${{ needs.authorize.outputs.pr_run_id }}",
        )
        self.assertFalse(rerun["concurrency"]["cancel-in-progress"])
        self.assertEqual(rerun["concurrency"]["queue"], "max")
        self.assertEqual(rerun["permissions"]["statuses"], "write")
        for name in ("publish-pr", "publish"):
            with self.subTest(job=name):
                self.assertEqual(self.lock_key(name, 123), "ci-status-123")
                self.assertEqual(
                    self.publisher.jobs[name]["permissions"]["statuses"], "write"
                )

    def test_resolver_cannot_admit_every_community_event(self):
        self.publisher.jobs["resolve"]["if"] = (
            "github.event_name == 'workflow_run' || (github.event_name == "
            "'workflow_dispatch' && github.ref == 'refs/heads/main')"
        )

        errors = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(
            any("resolve must enforce" in error for error in errors), errors
        )

    def test_trigger_run_lock_cannot_replace_qualification_lock(self):
        job = self.publisher.jobs["publish"]
        job["concurrency"] = {
            "group": "ci-status-${{ github.event.workflow_run.id }}",
            "cancel-in-progress": False,
            "queue": "max",
        }
        errors = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("serialize" in error for error in errors), errors)

    def test_resolver_cannot_gain_a_write_token(self):
        job = self.publisher.jobs.setdefault("resolve", {})
        job["permissions"] = {"contents": "read", "checks": "write"}
        errors = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("resolver" in error for error in errors), errors)

    def test_normal_and_community_publication_share_serialized_pending_state(self):
        client = PendingClient()
        sibling = client.event()
        normal = {"action": "completed", "workflow_run": copy.deepcopy(client.run)}
        keys = [
            self.lock_key("publish", resolve_event(client, REPO, sibling)["run_id"]),
            self.lock_key("publish-pr", normal["workflow_run"]["id"]),
        ]
        self.assertEqual(keys, ["ci-status-101", "ci-status-101"])
        locks = {key: threading.Lock() for key in keys}
        start = threading.Barrier(2)

        def execute(index):
            start.wait(timeout=5)
            with locks[keys[index]]:
                if index == 0:
                    before = len(client.status_writes)
                    result = recover_pending(client, REPO, sibling, 101)
                    self.assertEqual(len(client.status_writes), before)
                    return result
                return reconcile(client, REPO, 101)

        with (
            mock.patch("subprocess.run", side_effect=AssertionError("unmocked CLI")),
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(pool.map(execute, (0, 1)))
        self.assertEqual(len(results), 2)
        self.assertEqual([row["state"] for row in client.status_writes], ["pending"])
        self.assertEqual(client.post_attempts, 1)
        self.assertEqual(
            len([c for c in client.checks if c["status"] == "in_progress"]), 1
        )
        self.assertFalse(any(c.get("conclusion") == "success" for c in client.checks))

    def test_community_waiter_does_not_displace_terminal_publication(self):
        for verdict in ("success", "failure"):
            with self.subTest(verdict=verdict):
                client = PendingClient()
                client.visible = {80}
                client.write = super(PendingClient, client).write
                client.run.update(status="in_progress", conclusion=None)
                started = {
                    "action": "in_progress",
                    "workflow_run": copy.deepcopy(client.run),
                }
                running_job = invoke(client, started, resolve=True)
                self.assertEqual(invoke(client, started).get("state"), "running")

                client.run.update(status="completed", conclusion=verdict)
                if verdict == "success":
                    client.plan["deferred_verifications"] = []
                    client.plan["required_verification_ids"] = client.plan[
                        "expected_verification_ids"
                    ]
                    client.refresh_report()
                terminal = {
                    "action": "completed",
                    "workflow_run": copy.deepcopy(client.run),
                }
                community = client.event()
                jobs = [
                    running_job,
                    invoke(client, terminal, resolve=True),
                    invoke(client, community, resolve=True),
                ]
                self.assertEqual(
                    [job["mode"] for job in jobs], ["publish", "publish", "pending"]
                )
                groups = [
                    self.lock_key("publish-pr", started["workflow_run"]["id"]),
                    self.lock_key("publish-pr", terminal["workflow_run"]["id"]),
                    self.lock_key("publish", jobs[2]["run_id"]),
                ]
                self.assertEqual(groups, ["ci-status-101"] * 3)

                # While the first publisher owns the group, the default
                # Actions queue replaces its prior waiter with the new job.
                waiting = [terminal, community]
                if self.publisher.jobs["publish"]["concurrency"].get("queue") != "max":
                    waiting = waiting[-1:]
                results = [invoke(client, event) for event in waiting]

                self.assertEqual(
                    [check["conclusion"] for check in client.checks],
                    [verdict],
                    "Community recovery displaced the terminal publisher",
                )
                self.assertEqual(results[0].get("state"), verdict)
                self.assertIn("ignored", results[1])

    def test_status_publisher_rejects_replacing_waiting_jobs(self):
        for name in ("publish-pr", "publish"):
            for queue in (None, "single"):
                with self.subTest(job=name, queue=queue):
                    workflows = copy.deepcopy(self.workflows)
                    concurrency = workflows["ci-status.yml"].jobs[name]["concurrency"]
                    if queue is None:
                        concurrency.pop("queue", None)
                    else:
                        concurrency["queue"] = queue
                    errors = []
                    validate_status_publisher(workflows, errors)
                    self.assertTrue(
                        any("keep queued" in error for error in errors), errors
                    )

    def test_pull_request_publisher_requires_resolution_under_the_same_lock(self):
        step = self.publisher.jobs["publish-pr"]["steps"][-1]
        step["if"] = "steps.qualification.outputs.run_id != ''"
        errors = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("identity and mode for its lock" in e for e in errors))

    def test_current_workflows_preserve_the_pr_gate_contract(self) -> None:
        errors: list[str] = []
        validate_pr_contract(self.workflows, errors)
        self.assertEqual(errors, [])

    def test_gates_cannot_block_cancellation_or_hide_failed_prerequisites(self) -> None:
        for filename, job in (("pr.yml", "pr-gate"), ("ci.yml", "gate")):
            for condition in ("always()", "success()", None):
                with self.subTest(workflow=filename, condition=condition):
                    workflows = copy.deepcopy(self.workflows)
                    gate = workflows[filename].jobs[job]
                    gate["if"] = condition
                    errors: list[str] = []
                    validate_pr_contract(workflows, errors)
                    self.assertTrue(
                        any(
                            f"{filename} gate must reconcile failures without blocking workflow cancellation"
                            in error
                            for error in errors
                        ),
                        errors,
                    )

    def test_shared_image_producer_uses_no_layer_cache(self) -> None:
        for key in ("cache-from", "cache-to"):
            for value in ("type=gha,scope=vllm-sr-candidate", "${{ env.LAYER_CACHE }}"):
                with self.subTest(key=key, value=value):
                    workflows = copy.deepcopy(self.workflows)
                    build = next(
                        step
                        for step in workflows["build-artifacts.yml"].jobs["image"][
                            "steps"
                        ]
                        if "docker/build-push-action" in step.get("uses", "")
                    )
                    build["with"][key] = value
                    errors: list[str] = []
                    validate_pr_contract(workflows, errors)
                    self.assertIn(
                        "shared image producer must not import or export a layer cache",
                        errors,
                    )

    def test_queue_checks_on_drafts_that_need_no_new_ci_request(self) -> None:
        errors: list[str] = []
        validate_mergify_contract(errors)
        self.assertEqual(errors, [])
        config = yaml.safe_load((ROOT / ".mergify.yml").read_text(encoding="utf-8"))
        del config["queue_rules"][0]["max_checks_retries"]
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / ".mergify.yml").write_text(yaml.safe_dump(config))
            with mock.patch.object(workflow_policy_validation, "REPO_ROOT", Path(root)):
                validate_mergify_contract(errors)
        self.assertTrue(any("max_checks_retries" in error for error in errors), errors)

    def test_publisher_checkout_cannot_use_the_observed_pr_head(self) -> None:
        for name in ("publish-pr", "publish"):
            with self.subTest(job=name):
                workflows = copy.deepcopy(self.workflows)
                checkout = workflows["ci-status.yml"].jobs[name]["steps"][0]
                checkout["with"]["ref"] = "${{ github.event.workflow_run.head_sha }}"
                errors: list[str] = []
                validate_status_publisher(workflows, errors)
                self.assertTrue(
                    any("never check out" in error for error in errors), errors
                )

    def test_publisher_cannot_execute_untrusted_artifacts(self) -> None:
        self.publisher.jobs["publish"]["steps"].append(
            {"run": "python3 artifact/run.py"}
        )
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(
            any("trusted publisher entrypoint" in error for error in errors)
        )

    def test_publisher_dispatch_cannot_run_a_topic_branch(self) -> None:
        self.publisher.jobs["publish"]["if"] = "always()"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("trusted main code" in error for error in errors))

    def test_publisher_observes_reruns_not_only_initial_requests(self) -> None:
        self.publisher.events["workflow_run"]["types"] = ["requested", "completed"]
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("reruns and completions" in error for error in errors))

    def test_publisher_cannot_gain_source_write_permission(self) -> None:
        self.publisher.jobs["publish"]["permissions"]["contents"] = "write"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("token permissions" in error for error in errors))

    def test_issue_write_does_not_replace_pr_feedback_write_permission(self) -> None:
        self.publisher.jobs["publish"]["permissions"]["pull-requests"] = "read"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("token permissions" in error for error in errors))

    def test_bootstrap_probe_cannot_use_the_checkout_tree(self) -> None:
        steps = self.workflows["ci-changes.yml"].jobs["plan"]["steps"]
        probe = next(step for step in steps if step.get("id") == "publisher")
        probe["run"] = probe["run"].replace("$base:", "HEAD:")
        errors: list[str] = []
        validate_publisher_bootstrap(self.workflows, errors)
        self.assertTrue(any("actual merge first parent" in error for error in errors))

    def test_gate_requires_publisher_readiness_before_deferring(self) -> None:
        gate = copy.deepcopy(self.workflows["ci.yml"].jobs["gate"])
        step = next(
            step for step in gate["steps"] if "check_ci_gate.py" in step.get("run", "")
        )
        step["run"] = step["run"].replace(
            'if [[ "$PUBLISHER_READY" == true ]]; then flags+=(--allow-deferred); fi',
            "flags+=(--allow-deferred)",
        )
        errors: list[str] = []
        validate_gate_transport(gate, errors)
        self.assertTrue(
            any("only with a trusted base publisher" in error for error in errors)
        )

    def test_failed_gate_outcome_must_still_be_uploaded(self) -> None:
        gate = self.workflows["ci.yml"].jobs["gate"]
        upload = next(
            step
            for step in gate["steps"]
            if step.get("with", {}).get("name") == "ci-gate"
        )
        upload.pop("if")
        errors: list[str] = []
        validate_gate_transport(gate, errors)
        self.assertTrue(any("always publish" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
