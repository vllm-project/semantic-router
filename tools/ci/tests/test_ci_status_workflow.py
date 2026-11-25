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

    def test_ci_execution_reads_only_the_current_reconciled_revision(self):
        shared = self.workflows["ci.yml"]
        gate = shared.jobs["gate"]
        reconcile_step = next(
            step for step in gate["steps"] if "check_ci_gate.py" in step.get("run", "")
        )
        execution = self.workflows["pr.yml"].jobs["pr-gate"]["steps"][0]
        planner = self.workflows["ci-changes.yml"]
        plan_job = planner.jobs["plan"]
        self.assertEqual(
            plan_job["outputs"]["plan_run"], "${{ steps.plan.outputs.plan_run }}"
        )
        for output in ("plan_run", "strict_ci_execution"):
            self.assertEqual(
                planner.events["workflow_call"]["outputs"][output]["value"],
                "${{ jobs.plan.outputs." + output + " }}",
            )
        self.assertEqual(
            reconcile_step["env"]["CI_PLAN_RUN"], "${{ needs.plan.outputs.plan_run }}"
        )
        self.assertEqual(
            shared.events["workflow_call"]["outputs"]["strict_ci_execution"]["value"],
            "${{ jobs.plan.outputs.strict_ci_execution }}",
        )
        self.assertEqual(
            execution["env"]["STRICT_CI_EXECUTION"],
            "${{ needs.ci.outputs.strict_ci_execution }}",
        )
        head = "${{ github.event.pull_request.head.sha }}"
        self.assertEqual(reconcile_step["env"]["PR_HEAD_SHA"], head)
        self.assertEqual(
            gate["outputs"]["qualified_revision"],
            "${{ steps." + reconcile_step["id"] + ".outputs.qualified_revision }}",
        )
        self.assertEqual(
            shared.events["workflow_call"]["outputs"]["qualified_revision"]["value"],
            "${{ jobs.gate.outputs.qualified_revision }}",
        )
        self.assertEqual(execution["env"]["PR_HEAD_SHA"], head)
        self.assertEqual(
            execution["env"]["QUALIFIED_REVISION"],
            "${{ needs.ci.outputs.qualified_revision }}",
        )
        self.assertIn(
            '"$PR_HEAD_SHA:$GITHUB_SHA:$GITHUB_RUN_ID:$GITHUB_RUN_ATTEMPT"',
            execution["run"],
        )

    def test_event_resolution_precedes_one_shared_qualification_lock(self):

        client = PendingClient()
        sibling = invoke(client, resolve=True)
        normal = invoke(
            client,
            {"action": "completed", "workflow_run": copy.deepcopy(client.run)},
            resolve=True,
        )
        lock = (
            self.publisher.jobs["publish"]
            .get("concurrency", self.publisher.data.get("concurrency", {}))
            .get("group", "")
        )
        keys = []
        for event_id, resolved in [(202, sibling), (101, normal)]:
            keys.append(
                lock.replace(
                    "${{ needs.resolve.outputs.run_id }}", str(resolved.get("run_id"))
                ).replace(
                    "${{ github.event.workflow_run.id || inputs.run_id }}",
                    str(event_id),
                )
            )
        self.assertEqual(keys, ["ci-status-101", "ci-status-101"])

    def test_resolver_has_no_write_permission(self):
        job = self.publisher.jobs.get("resolve")
        self.assertIsNotNone(job)
        self.assertEqual(set(job["permissions"].values()), {"read"})

    def test_commit_status_rerun_shares_the_publisher_lock_for_the_authorized_run(self):
        rerun = self.workflows["manual-ci.yml"].jobs["rerun"]
        group = self.publisher.jobs["publish"]["concurrency"]["group"]
        self.assertEqual(
            rerun["concurrency"]["group"],
            "ci-status-${{ needs.authorize.outputs.pr_run_id }}",
        )
        self.assertEqual(
            group.replace("${{ needs.resolve.outputs.run_id }}", "123"),
            "ci-status-123",
        )
        self.assertFalse(rerun["concurrency"]["cancel-in-progress"])
        self.assertEqual(rerun["concurrency"]["queue"], "max")
        self.assertEqual(rerun["permissions"]["statuses"], "write")
        self.assertEqual(
            self.publisher.jobs["publish"]["permissions"]["statuses"], "write"
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
        identities = [resolve_event(client, REPO, event) for event in (sibling, normal)]
        template = self.publisher.jobs["publish"]["concurrency"]["group"]
        keys = [
            template.replace("${{ needs.resolve.outputs.run_id }}", str(i["run_id"]))
            for i in identities
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
        concurrency = self.publisher.jobs["publish"]["concurrency"]
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
                    concurrency["group"].replace(
                        "${{ needs.resolve.outputs.run_id }}", str(job["run_id"])
                    )
                    for job in jobs
                ]
                self.assertEqual(groups, ["ci-status-101"] * 3)

                # While the first publisher owns the group, the default
                # Actions queue replaces its prior waiter with the new job.
                waiting = [terminal, community]
                if concurrency.get("queue", "single") == "single":
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
        for queue in (None, "single"):
            with self.subTest(queue=queue):
                workflows = copy.deepcopy(self.workflows)
                concurrency = workflows["ci-status.yml"].jobs["publish"]["concurrency"]
                if queue is None:
                    concurrency.pop("queue", None)
                else:
                    concurrency["queue"] = queue
                errors = []
                validate_status_publisher(workflows, errors)
                self.assertTrue(any("keep queued" in error for error in errors), errors)

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

    def test_queue_requires_both_checks_from_github_actions(self) -> None:
        config = yaml.safe_load((ROOT / ".mergify.yml").read_text(encoding="utf-8"))
        for section in ("queue_conditions", "merge_conditions"):
            for context in ("PR Qualification", "CI Execution"):
                qualified = f"check-success = @github-actions/{context}"
                replacements = (
                    f"check-success = {context}",
                    {"or": [qualified, "check-success = PR Gate"]},
                )
                for replacement in replacements:
                    with self.subTest(section=section, replacement=replacement):
                        modified = copy.deepcopy(config)
                        conditions = modified["queue_rules"][0][section]
                        conditions[conditions.index(qualified)] = replacement
                        with tempfile.TemporaryDirectory() as root:
                            (Path(root) / ".mergify.yml").write_text(
                                yaml.safe_dump(modified), encoding="utf-8"
                            )
                            errors: list[str] = []
                            with mock.patch.object(
                                workflow_policy_validation, "REPO_ROOT", Path(root)
                            ):
                                validate_mergify_contract(errors)
                        self.assertIn(
                            f".mergify.yml: {section} must require @github-actions/{context}",
                            errors,
                        )

    def test_queue_rejects_a_required_legacy_gate(self) -> None:
        config = yaml.safe_load((ROOT / ".mergify.yml").read_text(encoding="utf-8"))
        config["queue_rules"][0]["merge_conditions"].append(
            "check-success = @github-actions/PR Gate"
        )
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / ".mergify.yml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            errors: list[str] = []
            with mock.patch.object(workflow_policy_validation, "REPO_ROOT", Path(root)):
                validate_mergify_contract(errors)
        self.assertEqual(
            errors, [".mergify.yml: merge_conditions retains compatibility contexts"]
        )

    def test_publisher_checkout_cannot_use_the_observed_pr_head(self) -> None:
        checkout = self.publisher.jobs["publish"]["steps"][0]
        checkout["with"]["ref"] = "${{ github.event.workflow_run.head_sha }}"
        errors: list[str] = []
        validate_status_publisher(self.workflows, errors)
        self.assertTrue(any("never check out" in error for error in errors), errors)

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
