"""Regression cases for #5455's same-parent merge regeneration boundary."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import manual_ci
import pr_ci_status
import test_manual_ci as authorization
import test_pr_ci_status as status

REPLACEMENT = "e" * 40
TREE = "1" * 40
OTHER_TREE = "2" * 40
UNAVAILABLE = "planned_merge_unavailable"


def commit(base, head, tree=TREE):
    return {"parents": [{"sha": base}, {"sha": head}], "tree": {"sha": tree}}


class AuthorizationClient(authorization.FakeGitHub):
    def __init__(self):
        super().__init__()
        self.commits = {
            authorization.MERGE: commit(authorization.BASE, authorization.HEAD),
            REPLACEMENT: commit(authorization.BASE, authorization.HEAD),
        }

    def request(self, endpoint, **kwargs):
        if "/git/commits/" in endpoint:
            self.calls.append(endpoint)
            value = self.commits[endpoint.rsplit("/", 1)[-1]]
            if isinstance(value, manual_ci.ManualCIError):
                raise value
            return copy.deepcopy(value)
        return super().request(endpoint, **kwargs)


class EquivalentAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.client = AuthorizationClient()
        self.environment = patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_REPOSITORY": authorization.REPO,
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def authorize(self):
        event = {
            "action": "created",
            "repository": {"full_name": authorization.REPO},
            "sender": {"login": "maintainer"},
            "issue": {
                "number": 42,
                "pull_request": {"url": "https://api.github.com/pr"},
            },
            "comment": copy.deepcopy(self.client.comment),
        }
        with patch.dict(
            os.environ,
            {
                "GITHUB_EVENT_NAME": "issue_comment",
                "GITHUB_RUN_ID": "456",
                "GITHUB_SHA": authorization.TRUSTED,
            },
        ):
            return manual_ci.authorize(self.client, event)

    def resolve(self):
        return manual_ci.resolve(
            self.client,
            {"number": 42, "pull_request": {"head": {"sha": authorization.HEAD}}},
            source=authorization.MERGE,
            base=authorization.BASE,
        )

    def test_replacement_accepts_new_authorization_for_original_plan(self):
        self.client.pull["merge_commit_sha"] = REPLACEMENT
        try:
            record = self.authorize()
        except manual_ci.ManualCIError as error:
            self.fail(f"Equivalent merge rejected a current Plan: {error}")
        self.assertEqual(record["source_sha"], authorization.MERGE)
        self.assertEqual(record["base_sha"], authorization.BASE)
        self.assertEqual(record["pr_run_id"], 123)
        self.assertEqual(record["plan_sha256"], self.client.plan["plan_sha256"])
        self.client.add_authorization(record)
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )
        self.assertEqual(self.client.posts, [])

    def test_replacement_keeps_existing_authorization_consumable(self):
        record = self.authorize()
        self.client.add_authorization(record)
        self.client.pull["merge_commit_sha"] = REPLACEMENT
        try:
            manual_ci.rerun(self.client, record, wait_seconds=0)
        except manual_ci.ManualCIError as error:
            self.fail(f"Equivalent merge rejected an existing authorization: {error}")
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )

    def test_replacement_does_not_accept_wrong_or_malformed_planned_parents(self):
        for parents in (
            [{"sha": authorization.HEAD}, {"sha": authorization.BASE}],
            [{"sha": "f" * 40}, {"sha": authorization.HEAD}],
            [{"sha": authorization.BASE}, {"sha": "f" * 40}],
            [{"sha": authorization.BASE}],
            [
                {"sha": authorization.BASE},
                {"sha": authorization.HEAD},
                {"sha": REPLACEMENT},
            ],
        ):
            with self.subTest(parents=parents):
                self.client.pull["merge_commit_sha"] = authorization.MERGE
                self.client.commits[authorization.MERGE] = commit(
                    authorization.BASE, authorization.HEAD
                )
                record = self.authorize()
                self.client.artifacts = []
                self.client.add_authorization(record)
                self.client.pull["merge_commit_sha"] = REPLACEMENT
                self.client.commits[authorization.MERGE] = {"parents": parents}
                with self.assertRaises(manual_ci.ManualCIError):
                    self.authorize()
                self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)
                with self.assertRaises(manual_ci.ManualCIError):
                    manual_ci.rerun(self.client, record, wait_seconds=0)
                self.assertEqual(self.client.posts, [])

    def test_replacement_requires_matching_verifiable_trees(self):
        for source in (authorization.MERGE, REPLACEMENT):
            for tree in (
                None,
                {},
                {"sha": None},
                {"sha": "invalid"},
                {"sha": 1},
                {"sha": OTHER_TREE},
            ):
                with self.subTest(source=source, tree=tree):
                    self.client = AuthorizationClient()
                    record = self.authorize()
                    self.client.add_authorization(record)
                    self.client.pull["merge_commit_sha"] = REPLACEMENT
                    if tree is None:
                        self.client.commits[source].pop("tree")
                    else:
                        self.client.commits[source]["tree"] = tree
                    with self.assertRaises(manual_ci.ManualCIError):
                        self.authorize()
                    self.assertEqual(self.resolve(), manual_ci.NO_AUTHORIZATION)
                    with self.assertRaises(manual_ci.ManualCIError):
                        manual_ci.rerun(self.client, record, wait_seconds=0)
                    self.assertEqual(self.client.posts, [])

    def test_ancestor_base_change_does_not_require_equal_trees(self):
        new_base = "d" * 40
        self.client.base_ref["object"]["sha"] = new_base
        self.client.pull["base"]["sha"] = new_base
        self.client.pull["merge_commit_sha"] = REPLACEMENT
        self.client.commits[REPLACEMENT] = commit(
            new_base, authorization.HEAD, OTHER_TREE
        )
        record = self.authorize()
        self.assertEqual(record["base_sha"], authorization.BASE)
        self.assertEqual(record["source_sha"], authorization.MERGE)
        self.client.add_authorization(record)
        self.assertEqual(
            self.resolve(), {"authorized": True, "scopes": ["affected"], "full": False}
        )
        manual_ci.rerun(self.client, record, wait_seconds=0)
        self.assertEqual(
            self.client.posts, [f"repos/{authorization.REPO}/actions/runs/123/rerun"]
        )

    def test_replacement_cannot_bypass_plan_digest(self):
        self.client.pull["merge_commit_sha"] = REPLACEMENT
        self.client.plan["plan_sha256"] = "invalid"
        with self.assertRaises(manual_ci.ManualCIError):
            self.authorize()
        self.assertEqual(self.client.posts, [])

    def test_replacement_does_not_fall_back_to_a_valid_older_plan(self):
        self.client.pull["merge_commit_sha"] = REPLACEMENT
        try:
            record = self.authorize()
        except manual_ci.ManualCIError as error:
            self.fail(f"The older Plan must be valid for this control: {error}")
        self.assertEqual(record["pr_run_id"], 123)
        self.client.pr_runs = [
            self.client.pr_run,
            {**self.client.pr_run, "id": 124, "created_at": "2026-10-18T10:00:30Z"},
        ]
        with self.assertRaises(manual_ci.ManualCIError):
            self.authorize()
        self.assertEqual(self.client.posts, [])

    def test_unreadable_original_merge_has_explicit_authorization_failure(self):
        record = self.authorize()
        self.client.add_authorization(record)
        self.client.pull["merge_commit_sha"] = REPLACEMENT
        self.client.commits[authorization.MERGE] = manual_ci.ManualCIError(
            "untrusted original-object payload", http_status=404
        )
        for operation in (
            self.authorize,
            self.resolve,
            lambda: manual_ci.rerun(self.client, record, wait_seconds=0),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(manual_ci.ManualCIError) as caught:
                    operation()
                self.assertEqual(caught.exception.reason, UNAVAILABLE)
                self.assertNotIn(
                    "untrusted original-object payload", str(caught.exception)
                )
        self.assertEqual(self.client.posts, [])


class PublisherClient(status.Client):
    def __init__(self):
        super().__init__()
        self.commits = {
            status.MERGE: commit(status.BASE, status.HEAD),
            REPLACEMENT: commit(status.BASE, status.HEAD),
        }

    def request(self, endpoint, **kwargs):
        if "/git/commits/" in endpoint:
            value = self.commits[endpoint.rsplit("/", 1)[-1]]
            if isinstance(value, manual_ci.ManualCIError):
                raise value
            return copy.deepcopy(value)
        return super().request(endpoint, **kwargs)


def qualification(client, state):
    if state != "awaiting-approval":
        client.plan.update(
            expected_verification_ids=["quality", "e2e.dashboard"],
            deferred_verifications=[],
            deferred_images=[],
        )
        client.refresh_report()
    if state == "failure":
        client.run["conclusion"] = "failure"
        client.jobs[-1]["conclusion"] = "failure"


class EquivalentPublisherTests(unittest.TestCase):
    def test_replacement_preserves_pending_passed_and_failed_qualification(self):
        for state in ("awaiting-approval", "success", "failure"):
            with self.subTest(state=state):
                client = PublisherClient()
                qualification(client, state)
                client.pull["merge_commit_sha"] = REPLACEMENT
                result = pr_ci_status.reconcile(client, status.REPO, 101)
                self.assertEqual(result["state"], state)
                self.assertEqual(
                    result["check"].get("conclusion"),
                    None if state == "awaiting-approval" else state,
                )

    def test_unreadable_source_is_identifiable_and_cannot_publish_success(self):
        for state in ("awaiting-approval", "success", "failure"):
            with self.subTest(state=state):
                client = PublisherClient()
                qualification(client, state)
                client.pull["merge_commit_sha"] = REPLACEMENT
                client.commits[status.MERGE] = manual_ci.ManualCIError(
                    "untrusted original-object payload", http_status=404
                )
                result = pr_ci_status.reconcile(client, status.REPO, 101)
                self.assertEqual(result["state"], "failure")
                self.assertEqual(result["check"]["conclusion"], "failure")
                self.assertIn(UNAVAILABLE, result.get("reason_codes", []))
                self.assertNotIn(
                    "untrusted original-object payload", json.dumps(result)
                )

    def test_equivalent_merge_changed_before_write_still_stops_publication(self):
        client = PublisherClient()
        qualification(client, "success")
        original = client.request
        reads = 0
        revalidation_read = 2

        def request(endpoint, **kwargs):
            nonlocal reads
            if endpoint.endswith("/pulls/3"):
                reads += 1
                if reads == revalidation_read:
                    client.pull["merge_commit_sha"] = REPLACEMENT
            return original(endpoint, **kwargs)

        client.request = request
        self.assertIn("ignored", pr_ci_status.reconcile(client, status.REPO, 101))
        self.assertEqual(client.writes, [])

    def test_unreadable_source_preview_is_write_free(self):
        client = PublisherClient()
        client.pull["merge_commit_sha"] = REPLACEMENT
        client.commits[status.MERGE] = manual_ci.ManualCIError(
            "missing", http_status=404
        )
        result = pr_ci_status.reconcile(client, status.REPO, 101, dry_run=True)
        self.assertEqual(result["check"]["conclusion"], "failure")
        self.assertIn(UNAVAILABLE, result.get("reason_codes", []))
        self.assertEqual(client.writes, [])


class DiagnosticClient(status.DiagnosticClient):
    def __init__(self):
        super().__init__()
        self.checks.pop()
        self.checks[0]["check_suite"]["id"] = 20
        self.commits = {
            status.MERGE: commit(status.BASE, status.HEAD),
            REPLACEMENT: commit(status.BASE, status.HEAD),
        }

    def request(self, endpoint, **kwargs):
        if "/git/commits/" in endpoint:
            value = self.commits[endpoint.rsplit("/", 1)[-1]]
            if isinstance(value, manual_ci.ManualCIError):
                raise value
            return copy.deepcopy(value)
        return super().request(endpoint, **kwargs)


class EquivalentDiagnosticTests(unittest.TestCase):
    def test_replacement_with_a_different_or_unverifiable_tree_is_not_current(self):
        for source in (status.MERGE, REPLACEMENT):
            for tree in ({"sha": OTHER_TREE}, {}):
                with self.subTest(source=source, tree=tree):
                    client = DiagnosticClient()
                    client.pull["merge_commit_sha"] = REPLACEMENT
                    client.commits[source]["tree"] = tree
                    result = pr_ci_status.diagnose(
                        client, status.REPO, 3, status.HEAD, 101, 1
                    )
                    self.assertEqual(result["state"], "source_stale")
                    self.assertFalse(result["evidence"]["current_source"])
                    self.assertFalse(result["actionable"])

    def test_original_source_disappearing_during_diagnosis_stays_unavailable(self):
        for missing_on in (2, 3):
            with self.subTest(missing_on=missing_on):
                client = DiagnosticClient()
                client.pull["merge_commit_sha"] = REPLACEMENT
                original = client.request
                reads = 0

                def request(
                    endpoint, *, original=original, missing_on=missing_on, **kwargs
                ):
                    nonlocal reads
                    if endpoint.endswith(f"/git/commits/{status.MERGE}"):
                        reads += 1
                        if reads >= missing_on:
                            raise manual_ci.ManualCIError("missing", http_status=404)
                    return original(endpoint, **kwargs)

                client.request = request
                result = pr_ci_status.diagnose(
                    client, status.REPO, 3, status.HEAD, 101, 1
                )
                self.assertEqual(result["state"], "evidence_unavailable")
                self.assertIn(UNAVAILABLE, result["reason_codes"])
                self.assertFalse(result["actionable"])

    def test_replacement_with_valid_source_is_healthy(self):
        client = DiagnosticClient()
        client.pull["merge_commit_sha"] = REPLACEMENT
        result = pr_ci_status.diagnose(client, status.REPO, 3, status.HEAD, 101, 1)
        self.assertEqual(result["state"], "healthy")
        self.assertTrue(result["evidence"]["current_source"])
        self.assertTrue(result["evidence"]["snapshot_current"])

    def test_unreadable_original_source_has_a_bounded_reason(self):
        client = DiagnosticClient()
        client.pull["merge_commit_sha"] = REPLACEMENT
        client.commits[status.MERGE] = manual_ci.ManualCIError(
            "untrusted original-object payload", http_status=404
        )
        result = pr_ci_status.diagnose(client, status.REPO, 3, status.HEAD, 101, 1)
        self.assertEqual(result["state"], "evidence_unavailable")
        self.assertIn(UNAVAILABLE, result["reason_codes"])
        self.assertFalse(result["actionable"])
        self.assertNotIn("untrusted original-object payload", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
