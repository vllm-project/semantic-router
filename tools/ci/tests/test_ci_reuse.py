"""Requested CI reuses evidence, never a success label alone."""

import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_ci_gate import evaluate_gate, load_receipts, receipt_artifact_names
from ci_plan import digest, github_outputs, make_plan
from ci_reuse import reuse_automatic_results, select_receipts
from execution_batches import scheduled_verifications
from test_ci_gate import completed

ENVIRONMENT = {"GITHUB_ACTIONS": "true", "CI_RUNNER_ENVIRONMENT": "github-hosted"}

# The artifact index whose archive carries the receipts in this fixture.
RECEIPT_INDEX = 2


def plans(paths=None):
    previous, receipts, _ = completed(
        paths or ["src/semantic-router/pkg/config/config.go"],
        require_approval=True,
        base_sha="b" * 40,
        execution_environment=ENVIRONMENT,
    )
    authorized = make_plan(
        previous["paths"],
        source_sha=previous["source_sha"],
        base_sha=previous["base_sha"],
        require_approval=True,
        authorization={"authorized": True, "scopes": ["affected"]},
    )
    return previous, authorized, receipts


def check_select_only_identical_successful_automatic_receipts():
    previous, plan, receipts = plans()
    assert len(receipts) > 1
    assert select_receipts(plan, previous, receipts) == receipts
    assert set(plan["expected_verification_ids"]) > {r["id"] for r in receipts}
    assert select_receipts(plan, previous, receipts[1:]) == receipts[1:]
    failed = copy.deepcopy(receipts)
    failed[0]["result"] = "failed"
    assert select_receipts(plan, previous, failed) == receipts[1:]
    assert select_receipts(plan, previous, [*receipts, receipts[0]]) == receipts[1:]


def check_changed_identity_is_not_reused(field):
    previous, plan, receipts = plans()
    if field in {"source_sha", "base_sha"}:
        plan[field] = "c" * 40
        assert select_receipts(plan, previous, receipts) == []
    else:
        plan["verifications"][0][field] = "changed"
        assert receipts[0] not in select_receipts(plan, previous, receipts)


def check_invalid_evidence_runs_again(damage):
    previous, plan, receipts = plans()
    damaged = copy.deepcopy(receipts)
    receipt = damaged[0]
    if damage == "local":
        receipt["execution"]["runner_environment"] = "local"
    elif damage == "digest":
        receipt["evidence_sha256"] = "0" * 64
    elif damage == "source":
        receipt["source_sha"] = "c" * 40
    elif damage == "contract":
        receipt["contract_sha256"] = "0" * 64
    else:
        field = "cases" if "cases" in receipt["evidence"] else "checks"
        if damage == "failed-case":
            receipt["evidence"][field][0]["status"] = "failed"
        else:
            receipt["evidence"][field] = []
        receipt["evidence_sha256"] = digest(receipt["evidence"])
    assert select_receipts(plan, previous, damaged) == receipts[1:]


def archive(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        for name, value in files.items():
            output.writestr(name, json.dumps(value))
    return buffer.getvalue()


def transport(tmp_path, paths=None):
    previous, plan, receipts = plans(paths)
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"pull_request": {"head": {"sha": "d" * 40}}}))
    env = {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": plan["source_sha"],
        "GITHUB_EVENT_PATH": str(event_path),
    }
    run = {
        "id": 42,
        "run_attempt": 1,
        "status": "completed",
        "event": "pull_request",
        "repository": {"full_name": "owner/repo"},
        "path": ".github/workflows/pr.yml",
        "workflow_id": 7,
        "head_sha": "d" * 40,
    }
    artifacts = [
        {"id": 1, "name": "ci-plan", "size_in_bytes": 1000, "workflow_run": {"id": 42}}
    ]
    blobs = {1: archive({"plan.json": previous})}
    # Put receipts in the physical containers from the previous plan.
    for index, name in enumerate(sorted(receipt_artifact_names(previous)), 2):
        artifacts.append(
            {
                "id": index,
                "name": name,
                "size_in_bytes": 1000,
                "workflow_run": {"id": 42},
            }
        )
        # One container is enough for this fixture; the real gate still checks
        # every receipt's contract rather than trusting its filename.
        blobs[index] = archive(
            {f"{r['id']}.json": r for r in receipts} if index == RECEIPT_INDEX else {}
        )
    client = Mock()

    def request(endpoint):
        if "/attempts/" in endpoint:
            return run
        if endpoint.endswith("workflows/pr.yml"):
            return {"id": 7}
        return {"artifacts": artifacts}

    client.request.side_effect = request
    client.raw.side_effect = lambda endpoint: blobs[int(endpoint.split("/")[-2])]
    return plan, receipts, env, run, client


def check_authorized_attempt_dispatch_and_gate_reconcile_reused_results(tmp_path):
    plan, receipts, env, _, client = transport(tmp_path)
    output = tmp_path / "ci-result-reused"
    reuse_automatic_results(plan, output, client=client, environ=env)
    assert set(plan["reused_verification_ids"]) == {r["id"] for r in receipts}
    assert all(r["execution"] == "approved" for r in scheduled_verifications(plan))
    assert json.loads(github_outputs(plan)["quality"]) == []
    assert load_receipts(tmp_path, plan) == sorted(receipts, key=lambda r: r["id"])
    # Finish the newly authorized work, then reconcile the complete inventory.
    finished, all_receipts, builds = completed(
        plan["paths"],
        require_approval=True,
        base_sha=plan["base_sha"],
        authorization={"authorized": True, "scopes": ["affected"]},
        execution_environment=ENVIRONMENT,
    )
    plan["image_sources"] = finished["image_sources"]
    plan["plan_sha256"] = digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    jobs = {name: {"result": "success"} for name in plan["expected_dispatch_jobs"]}
    jobs["quality"] = {"result": "skipped"}
    verdict = evaluate_gate(
        plan, all_receipts, builds=builds, jobs=jobs, require_actions=True
    )
    assert verdict.passed, verdict.errors
    assert set(verdict.required) == set(finished["required_verification_ids"])
    assert not evaluate_gate(
        plan, all_receipts[1:], builds=builds, jobs=jobs, require_actions=True
    ).passed


def check_foreign_or_unfinished_producer_cannot_supply_receipts(tmp_path, field, value):
    plan, _, env, run, client = transport(tmp_path)
    run[field] = value
    reuse_automatic_results(plan, tmp_path / "reused", client=client, environ=env)
    assert "reused_verification_ids" not in plan
    client.raw.assert_not_called()


def check_missing_artifacts_falls_back_to_running(tmp_path):
    plan, _, env, _, client = transport(tmp_path)
    client.raw.side_effect = ValueError("expired artifact")
    reuse_automatic_results(plan, tmp_path / "reused", client=client, environ=env)
    assert "reused_verification_ids" not in plan


class ReuseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_selection(self):
        check_select_only_identical_successful_automatic_receipts()

    def test_changed_identity(self):
        for field in (
            "source_sha",
            "base_sha",
            "contract_sha256",
            "inventory",
            "execution_target_sha256",
        ):
            with self.subTest(field=field):
                check_changed_identity_is_not_reused(field)

    def test_invalid_evidence(self):
        for damage in (
            "local",
            "digest",
            "failed-case",
            "source",
            "contract",
            "missing-case",
        ):
            with self.subTest(damage=damage):
                check_invalid_evidence_runs_again(damage)

    def test_dispatch_and_gate(self):
        check_authorized_attempt_dispatch_and_gate_reconcile_reused_results(self.root)

    def test_producer_trust(self):
        for field, value in (
            ("head_sha", "e" * 40),
            ("path", ".github/workflows/other.yml"),
            ("repository", {"full_name": "attacker/repo"}),
            ("workflow_id", 8),
            ("event", "workflow_dispatch"),
            ("status", "in_progress"),
            ("run_attempt", 2),
        ):
            with self.subTest(field=field):
                check_foreign_or_unfinished_producer_cannot_supply_receipts(
                    self.root, field, value
                )

    def test_failed_component_is_scheduled_without_repeating_successes(self):
        plan, receipts, env, _, client = transport(self.root, ["tools/ci/ci_plan.py"])
        failed = next(r for r in receipts if r["id"] == "harness-tools")
        read = client.raw.side_effect

        def damaged_archive(endpoint):
            raw = read(endpoint)
            with zipfile.ZipFile(io.BytesIO(raw)) as source:
                files = {
                    name: json.loads(source.read(name)) for name in source.namelist()
                }
            for value in files.values():
                if value.get("id") == failed["id"]:
                    value["result"] = "failed"
            return archive(files)

        client.raw.side_effect = damaged_archive
        reuse_automatic_results(
            plan, self.root / "ci-result-reused", client=client, environ=env
        )
        assert failed["id"] not in plan["reused_verification_ids"]
        automatic = [
            r["id"]
            for r in scheduled_verifications(plan)
            if r["execution"] == "automatic"
        ]
        assert automatic == [failed["id"]]
        assert any(
            failed["id"] in [r["id"] for r in b["verifications"]]
            for b in plan["component_batches"]
        )
        assert "tools" in plan["expected_dispatch_jobs"]
        assert "ci-result-reused" in receipt_artifact_names(plan)
        # A later authorization may carry these receipts again without loading
        # the obsolete physical batches from the original automatic attempt.
        next_plan = make_plan(
            plan["paths"],
            source_sha=plan["source_sha"],
            base_sha=plan["base_sha"],
            require_approval=True,
            authorization={"authorized": True, "scopes": ["affected"]},
        )
        preserved = load_receipts(self.root, plan)
        assert (
            sorted(select_receipts(next_plan, plan, preserved), key=lambda r: r["id"])
            == preserved
        )

    def test_missing_artifacts(self):
        check_missing_artifacts_falls_back_to_running(self.root)
