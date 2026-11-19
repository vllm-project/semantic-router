"""Exercise the trusted CLI and pending recovery against a fake GitHub transport."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pr_ci_status
from pr_ci_pending_fixtures import PendingClient
from test_pr_ci_status import BASE, HEAD, MERGE, REPO


def invoke(client, event=None, *, resolve=False, arguments=()):
    event = client.event() if event is None else event
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        json.dump(event, f)
        filename = f.name
    output, errors = io.StringIO(), io.StringIO()
    argv = ["pr_ci_status.py", "--event", filename]
    if resolve:
        argv.append("--resolve-event")
    argv.extend(arguments)
    try:
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, GITHUB_REPOSITORY=REPO),
            patch.object(pr_ci_status, "StatusClient", return_value=client),
            patch("subprocess.run", side_effect=AssertionError("unmocked CLI")),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            try:
                code = pr_ci_status.main()
            except SystemExit as exc:
                code = exc.code
        text = output.getvalue().strip()
        return json.loads(text) if text else {"exit_code": code}
    finally:
        Path(filename).unlink()


class PendingRecoveryTests(unittest.TestCase):
    def test_nullable_output_text_is_not_recovery_provenance(self):
        for visible in (False, True):
            with self.subTest(visible=visible):
                client = PendingClient()
                client.checks[0]["output"]["text"] = None
                if visible:
                    client.visible.add(80)
                    client.write = super(PendingClient, client).write
                    event = {
                        "action": "completed",
                        "workflow_run": copy.deepcopy(client.run),
                    }
                    result = invoke(client, event)
                    self.assertEqual(result.get("state"), "awaiting-approval")
                else:
                    result = invoke(client)
                    self.assertEqual(result.get("recovery"), "restored")

    def test_non_string_output_text_cannot_authorize_recovery(self):
        for text in (False, 1, [], {}):
            with self.subTest(text=text):
                client = PendingClient()
                client.checks[0]["output"]["text"] = text
                invoke(client)
                self.assertEqual(client.writes, [])

    def test_new_hidden_terminal_during_receipt_collection_prevents_post(self):
        client = PendingClient()

        def complete_sibling(path, current):
            text = current.checks[0].get("output", {}).get("text") or ""
            if "/jobs?" in path and text.startswith(
                "semantic-router-pending:v1:claim:"
            ):
                current.checks.append(
                    {
                        **copy.deepcopy(current.checks[0]),
                        "id": 99,
                        "status": "completed",
                        "conclusion": "failure",
                        "output": {"text": None},
                    }
                )

        client.on_read = complete_sibling
        invoke(client)
        self.assertEqual(client.post_attempts, 0)
        self.assertEqual(client.checks[-1]["conclusion"], "failure")

    def test_normal_publication_cannot_regress_a_new_terminal(self):
        for boundary in ("comments", "latest", "target"):
            with self.subTest(boundary=boundary):
                client = PendingClient()
                client.visible.add(80)
                client.write = super(PendingClient, client).write

                def complete(path, current, boundary=boundary):
                    hit = (
                        (
                            boundary == "comments"
                            and path.startswith("issues/3/comments?")
                        )
                        or (boundary == "latest" and "filter=latest" in path)
                        or (boundary == "target" and path == "check-runs/80")
                    )
                    if hit:
                        current.checks[0].update(
                            status="completed", conclusion="failure"
                        )

                client.on_read = complete
                event = {
                    "action": "completed",
                    "workflow_run": copy.deepcopy(client.run),
                }
                invoke(client, event)
                self.assertEqual(client.writes, [])
                self.assertEqual(client.checks[0]["conclusion"], "failure")

    def test_live_repository_metadata_changes_do_not_invalidate_recovery(self):
        client = PendingClient()

        def metadata(path, current):
            if path == "actions/runs/101":
                count = current.reads[path]
                current.run["repository"]["stargazers_count"] = count
                current.run["head_repository"]["description"] = str(count)
                current.pull["head"]["repo"]["stargazers_count"] = count
                current.pull["base"]["repo"]["description"] = str(count)

        client.on_read = metadata
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        self.assertEqual(client.post_attempts, 1)

    def test_terminal_on_exact_write_target_blocks_creation_or_retirement(self):
        for after_creation in (False, True):
            with self.subTest(after_creation=after_creation):
                client = PendingClient()

                def complete(path, current, after_creation=after_creation):
                    text = current.checks[0].get("output", {}).get("text") or ""
                    if (
                        path == "check-runs/80"
                        and text.startswith("semantic-router-pending:v1:claim:")
                        and bool(current.post_attempts) == after_creation
                    ):
                        current.checks[0].update(
                            status="completed", conclusion="failure"
                        )

                client.on_read = complete
                invoke(client)
                self.assertEqual(client.post_attempts, int(after_creation))
                self.assertEqual(client.checks[0]["conclusion"], "failure")
                self.assertFalse(
                    any(w[1].get("conclusion") == "neutral" for w in client.writes)
                )

    def test_qualification_changed_during_collection_cannot_create_pending(self):
        for change in ("attempt", "newer", "status", "conclusion", "closed"):
            with self.subTest(change=change):
                client = PendingClient()

                def supersede(path, current, change=change):
                    text = current.checks[0].get("output", {}).get("text") or ""
                    if "/jobs?" in path and text.startswith(
                        "semantic-router-pending:v1:claim:"
                    ):
                        if change == "attempt":
                            current.run["run_attempt"] = 2
                        elif change == "newer":
                            current.newer_run = True
                        elif change == "status":
                            current.run["status"] = "in_progress"
                        elif change == "conclusion":
                            current.run["conclusion"] = "failure"
                        else:
                            current.pull["state"] = "closed"

                client.on_read = supersede
                invoke(client)
                self.assertEqual(client.post_attempts, 0)
                self.assertFalse(
                    any(w[1].get("conclusion") == "neutral" for w in client.writes)
                )

    def test_successful_community_maps_to_current_qualification_before_lock(self):
        client = PendingClient()
        result = invoke(client, resolve=True)
        self.assertEqual(result.get("run_id"), 101)
        self.assertEqual(result.get("mode"), "pending")
        self.assertEqual(client.writes, [])

    def test_same_attempt_hidden_pending_is_restored_without_authorizing(self):
        client = PendingClient()
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        self.assertEqual(client.run["run_attempt"], 1)
        current = [c for c in client.checks if c["id"] in client.visible]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["status"], "in_progress")
        self.assertIsNone(current[0].get("conclusion"))
        self.assertIn("/ci run", current[0]["output"]["summary"])
        self.assertEqual(client.post_attempts, 1)

    def test_repeated_event_reuses_replacement(self):
        client = PendingClient()
        invoke(client)
        invoke(client)
        self.assertEqual(client.post_attempts, 1)
        self.assertEqual(
            len([c for c in client.checks if c["status"] == "in_progress"]), 1
        )

    def test_visible_pending_does_not_create_or_modify_any_check(self):
        client = PendingClient()
        client.visible = {80}
        invoke(client)
        self.assertEqual(client.writes, [])

    def test_completed_current_verdict_blocks_even_if_hidden(self):
        for conclusion in ("success", "failure", "neutral"):
            for visible in (False, True):
                with self.subTest(conclusion=conclusion, visible=visible):
                    client = PendingClient()
                    client.checks[0].update(status="completed", conclusion=conclusion)
                    client.visible = {80} if visible else set()
                    invoke(client)
                    self.assertEqual(client.writes, [])

    def test_unconfirmed_visibility_never_repeats_post_or_retires_original(self):
        client = PendingClient()
        client.new_check_visible = False
        first = invoke(client)
        second = invoke(client)
        self.assertEqual(first.get("recovery"), "unconfirmed")
        self.assertEqual(second.get("recovery"), "unconfirmed")
        self.assertEqual(client.post_attempts, 1)
        self.assertEqual(client.checks[0]["status"], "in_progress")
        self.assertIsNone(client.checks[0]["conclusion"])

    def test_uncertain_post_outcome_is_not_repeated(self):
        client = PendingClient()
        client.post_error = subprocess.TimeoutExpired("gh", 60)
        invoke(client)
        invoke(client)
        self.assertEqual(client.post_attempts, 1)

    def test_real_failure_or_invalid_receipt_never_gets_repaired(self):
        for invalid in ("job", "missing", "attempt"):
            with self.subTest(invalid=invalid):
                client = PendingClient()
                if invalid == "job":
                    client.jobs.append({"id": 777, "conclusion": "failure"})
                elif invalid == "missing":
                    client.missing_report = True
                else:
                    client.report["run_attempt"] = 9
                invoke(client)
                self.assertEqual(client.writes, [])

    def test_fully_qualified_evidence_is_not_pending_repair(self):
        client = PendingClient()
        client.plan.update(
            expected_verification_ids=client.plan["required_verification_ids"],
            deferred_verifications=[],
        )
        client.refresh_report()
        invoke(client)
        self.assertEqual(client.writes, [])

    def test_foreign_checks_and_other_attempts_are_not_retired(self):
        client = PendingClient()
        protected = []
        for identity, override in enumerate(
            [
                {"app": {"id": 999}},
                {"external_id": "semantic-router-ci:4:101:1"},
                {"external_id": "semantic-router-ci:3:102:1"},
                {"external_id": "semantic-router-ci:3:101:2"},
                {"external_id": "12345678-1234-1234-1234-123456789abc"},
            ],
            90,
        ):
            check = {**copy.deepcopy(client.checks[0]), "id": identity, **override}
            client.checks.append(check)
            protected.append(copy.deepcopy(check))
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        for original in protected:
            self.assertEqual(
                next(c for c in client.checks if c["id"] == original["id"]), original
            )

    def test_closed_changed_or_superseded_identity_has_no_writes(self):
        for invalid in ("closed", "head", "newer", "Community_attempt"):
            with self.subTest(invalid=invalid):
                client = PendingClient()
                event = client.event()
                if invalid == "closed":
                    client.pull["state"] = "closed"
                elif invalid == "head":
                    client.pull["head"]["sha"] = "f" * 40
                elif invalid == "newer":
                    client.newer_run = True
                else:
                    client.community["run_attempt"] = 3
                invoke(client, event)
                self.assertEqual(client.writes, [])

    def test_repository_metadata_variation_does_not_change_source_identity(self):
        client = PendingClient()
        event = client.event()
        event["workflow_run"]["head_repository"]["description"] = "event-only metadata"
        result = invoke(client, event, resolve=True)
        self.assertEqual(result.get("run_id"), 101)

    def test_ambiguous_or_foreign_community_events_cannot_resolve(self):
        for invalid in ("ambiguous", "workflow", "repository", "failed", "running"):
            with self.subTest(invalid=invalid):
                client = PendingClient()
                if invalid == "ambiguous":
                    client.extra_pulls = [{**copy.deepcopy(client.pull), "number": 4}]
                elif invalid == "workflow":
                    client.community["workflow_id"] = 999
                elif invalid == "repository":
                    client.community["repository"] = {"full_name": "foreign/router"}
                elif invalid == "failed":
                    client.community["conclusion"] = "failure"
                else:
                    client.community["status"] = "in_progress"
                result = invoke(client, resolve=True)
                self.assertNotIn("run_id", result)
                self.assertEqual(client.writes, [])

    def test_terminal_appearing_after_claim_cannot_be_overwritten(self):
        client = PendingClient()

        def completed(path, current):
            if "filter=all" in path and current.checks[0].get("output", {}).get(
                "text", ""
            ).startswith("semantic-router-pending:v1:claim:"):
                current.checks[0].update(status="completed", conclusion="failure")

        client.on_read = completed
        invoke(client)
        self.assertEqual(client.post_attempts, 0)
        self.assertEqual(client.checks[0]["conclusion"], "failure")
        self.assertFalse(
            any(w[1].get("status") == "in_progress" for w in client.writes)
        )

    def test_delayed_visibility_finishes_existing_replacement(self):
        client = PendingClient()
        client.new_check_visible = False
        invoke(client)
        client.visible.add(client.checks[-1]["id"])
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        self.assertEqual(client.post_attempts, 1)
        self.assertEqual(client.checks[0]["conclusion"], "neutral")

    def test_later_sibling_attempt_can_repair_a_new_visibility_loss(self):
        client = PendingClient()
        invoke(client)
        client.visible.clear()
        client.community["run_attempt"] = 3
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        self.assertEqual(client.run["run_attempt"], 1)
        self.assertEqual(client.post_attempts, 2)
        self.assertEqual(
            len([c for c in client.checks if c["status"] == "in_progress"]), 1
        )

    def test_same_event_does_not_recreate_a_confirmed_then_hidden_replacement(self):
        client = PendingClient()
        invoke(client)
        client.visible.clear()
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "unconfirmed")
        self.assertEqual(client.post_attempts, 1)

    def test_claim_with_no_observed_post_result_does_not_retry_creation(self):
        client = PendingClient()
        client.post_before_error = subprocess.TimeoutExpired("gh", 60)
        invoke(client)
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "unconfirmed")
        self.assertEqual(client.post_attempts, 1)
        self.assertEqual(client.checks[0]["status"], "in_progress")

    def test_normal_publisher_does_not_duplicate_an_unconfirmed_recovery(self):
        client = PendingClient()
        client.new_check_visible = False
        invoke(client)
        event = {"action": "completed", "workflow_run": copy.deepcopy(client.run)}
        invoke(client, event)
        self.assertEqual(client.post_attempts, 1)

    def test_images_only_waiting_remains_pending(self):
        client = PendingClient()
        client.plan.update(
            required_verification_ids=["quality"],
            deferred_verifications=[],
            deferred_images=["router"],
        )
        client.refresh_report()
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        self.assertFalse(
            any(w[1].get("conclusion") in {"success", "failure"} for w in client.writes)
        )

    def test_hidden_terminal_after_inventory_prevents_creation(self):
        client = PendingClient()

        def completed(path, current):
            if (
                "filter=latest" in path
                and current.post_attempts == 0
                and current.checks[0]
                .get("output", {})
                .get("text", "")
                .startswith("semantic-router-pending:v1:claim:")
            ):
                current.checks[0].update(status="completed", conclusion="failure")

        client.on_read = completed
        invoke(client)
        self.assertEqual(client.post_attempts, 0)
        self.assertEqual(client.checks[0]["conclusion"], "failure")

    def test_hidden_terminal_before_retirement_is_not_neutralized(self):
        client = PendingClient()
        latest_after_creation = 0
        retirement_read = 2

        def completed(path, current):
            nonlocal latest_after_creation
            if "filter=latest" in path and current.post_attempts:
                latest_after_creation += 1
                if latest_after_creation == retirement_read:
                    current.checks[0].update(status="completed", conclusion="failure")

        client.on_read = completed
        invoke(client)
        self.assertEqual(client.checks[0]["conclusion"], "failure")
        self.assertFalse(
            any(w[1].get("conclusion") == "neutral" for w in client.writes)
        )

    def test_visibility_lost_during_retirement_is_not_reported_restored(self):
        client = PendingClient()

        def hide_after_retirement(endpoint, payload, current):
            if payload.get("conclusion") == "neutral":
                current.visible.clear()

        client.after_write = hide_after_retirement
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "unconfirmed")
        invoke(client)
        self.assertEqual(client.post_attempts, 1)

    def test_recorded_5745_pending_topology_keeps_qualification_attempt_one(self):
        case = json.loads(
            (
                Path(__file__).parent / "testdata/pr_gate_pending_snapshot.json"
            ).read_text(encoding="utf-8")
        )
        client = PendingClient()
        gate = copy.deepcopy(case["gate"])
        # Public topology/IDs are real; qualification identity/receipt is synthetic.
        gate.update(
            head_sha=client.run["head_sha"],
            external_id="semantic-router-ci:3:101:1",
            run_attempt=1,
        )
        client.checks = [gate]
        client.visible = set(case["latest_gate_ids"])
        client.plan.update(
            required_verification_ids=["quality", *case["deferred_verifications"]],
            deferred_verifications=case["deferred_verifications"],
        )
        client.refresh_report()
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")
        self.assertEqual(client.run["run_attempt"], 1)
        self.assertEqual(client.post_attempts, 1)
        self.assertTrue(all(c.get("conclusion") != "success" for c in client.checks))

    def test_pending_preview_is_write_free(self):
        client = PendingClient()
        result = invoke(client, arguments=("--dry-run",))
        self.assertEqual(result.get("recovery"), "eligible")
        self.assertEqual(client.writes, [])

    def test_resolved_run_cannot_be_overridden_by_a_stale_dispatch_value(self):
        client = PendingClient()
        invoke(client, arguments=("--mode", "pending", "--run-id", "999"))
        self.assertEqual(client.writes, [])

    def test_malformed_or_duplicate_claim_evidence_blocks_writes(self):
        for text in (
            "semantic-router-pending:v1:invalid",
            "semantic-router-pending:v1:claim:202:2:999",
        ):
            with self.subTest(text=text):
                client = PendingClient()
                client.checks[0]["output"]["text"] = text
                invoke(client)
                self.assertEqual(client.writes, [])

    def test_terminal_alongside_a_hidden_pending_prevents_repair(self):
        client = PendingClient()
        client.checks.append(
            {
                **copy.deepcopy(client.checks[0]),
                "id": 81,
                "status": "completed",
                "conclusion": "failure",
            }
        )
        invoke(client)
        self.assertEqual(client.writes, [])

    def test_unreadable_planned_source_cannot_be_recovered(self):
        client = PendingClient()
        request = client.request

        def unavailable(endpoint, **kwargs):
            if endpoint.endswith(f"git/commits/{MERGE}"):
                raise pr_ci_status.ManualCIError("unavailable", http_status=404)
            return request(endpoint, **kwargs)

        client.request = unavailable
        invoke(client)
        self.assertEqual(client.writes, [])

    def test_ancestor_base_still_allows_pending_recovery(self):
        client = PendingClient()
        client.main_sha = "d" * 40
        result = invoke(client)
        self.assertEqual(result.get("recovery"), "restored")

    def test_equivalent_merge_tree_proof_is_required_for_pending_recovery(self):
        for replacement_tree in ("e" * 40, "f" * 40, None):
            with self.subTest(replacement_tree=replacement_tree):
                client = PendingClient()
                replacement = "d" * 40
                client.pull["merge_commit_sha"] = replacement
                request = client.request

                def commits(
                    endpoint,
                    *,
                    replacement=replacement,
                    replacement_tree=replacement_tree,
                    request=request,
                    **kwargs,
                ):
                    if endpoint.endswith(f"git/commits/{MERGE}") or endpoint.endswith(
                        f"git/commits/{replacement}"
                    ):
                        tree = (
                            "e" * 40 if endpoint.endswith(MERGE) else replacement_tree
                        )
                        return {
                            "parents": [{"sha": BASE}, {"sha": HEAD}],
                            "tree": {"sha": tree},
                        }
                    return request(endpoint, **kwargs)

                client.request = commits
                result = invoke(client)
                if replacement_tree == "e" * 40:
                    self.assertEqual(result.get("recovery"), "restored")
                else:
                    self.assertEqual(client.writes, [])


if __name__ == "__main__":
    unittest.main()
