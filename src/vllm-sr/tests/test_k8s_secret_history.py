"""Rollback retention and garbage-collection tests for K8s Secrets."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _k8s_secret_history_helpers import (  # noqa: E402
    deployment_manifest,
    deployment_with_volumes,
)
from _k8s_test_helpers import k8s_backend, managed_secret_payload  # noqa: E402
from cli.k8s_secret_history import (  # noqa: E402
    parse_helm_history_revisions,
    parse_helm_manifest_secret_refs,
)
from cli.k8s_secret_quarantine import (  # noqa: E402
    MANAGED_SECRET_QUARANTINE_ANNOTATION,
    ManagedSecretCandidate,
)

GC_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
STALE_CREATED_AT = GC_NOW - timedelta(minutes=16)


def managed_candidates(*names: str) -> dict[str, ManagedSecretCandidate]:
    return {
        name: ManagedSecretCandidate(
            created_at=STALE_CREATED_AT,
            uid=f"uid-{name}",
            resource_version=f"rv-{name}",
        )
        for name in names
    }


def managed_secret_created_at(name: str, created_at: str | None) -> dict:
    payload = managed_secret_payload(name, "alpha", {})
    payload["metadata"]["creationTimestamp"] = created_at
    return payload


def annotate_delete_not_before(payload: dict, value: str) -> dict:
    payload["metadata"]["annotations"] = {MANAGED_SECRET_QUARANTINE_ANNOTATION: value}
    return payload


def kubernetes_timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def test_history_parser_accepts_only_positive_integer_revisions():
    assert parse_helm_history_revisions(
        json.dumps([{"revision": 2}, {"revision": 3}])
    ) == [2, 3]
    for payload in (
        "not-json",
        "{}",
        "[]",
        '[{"revision":"2"}]',
        '[{"revision":0}]',
    ):
        assert parse_helm_history_revisions(payload) is None


def test_unknown_volume_source_prevents_gc_deletion(monkeypatch):
    instance = k8s_backend()
    manifest = yaml.safe_dump(
        deployment_with_volumes(
            [{"name": "future", "futureStorage": {"secretRef": {"name": "live"}}}]
        )
    )
    retained = parse_helm_manifest_secret_refs(manifest)
    events = []
    assert retained is None
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [])
    monkeypatch.setattr(instance, "_helm_retained_secret_refs", lambda: retained)
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_candidates",
        lambda: events.append("list") or ["alpha-vsr-env-live"],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name, _candidate: events.append(("delete", name)) or True,
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert events == []


def test_csi_reference_protects_current_and_rollback_generations(monkeypatch):
    instance = k8s_backend()
    active = "alpha-vsr-env-active"
    rollback = "alpha-vsr-env-rollback"
    orphan = "alpha-vsr-env-orphan"
    manifest = yaml.safe_dump(
        deployment_with_volumes(
            [
                {
                    "name": "rollback-credentials",
                    "csi": {
                        "driver": "example.csi.invalid",
                        "nodePublishSecretRef": {"name": rollback},
                    },
                }
            ],
            env_secret=active,
        )
    )
    protected = parse_helm_manifest_secret_refs(manifest)
    deleted = []
    assert protected == {active, rollback}
    monkeypatch.setattr(instance, "_protected_secret_refs", lambda: protected)
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_candidates",
        lambda: managed_candidates(active, rollback, orphan),
    )
    monkeypatch.setattr(instance, "_managed_secret_gc_now", lambda: GC_NOW)
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name, _candidate: deleted.append(name) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == [orphan]


def test_gc_retains_current_and_every_rollback_revision(monkeypatch):
    instance = k8s_backend()
    retained = {"alpha-vsr-env-current", "alpha-vsr-env-rollback"}
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", lambda: retained)
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_candidates",
        lambda: managed_candidates(*sorted(retained), "alpha-vsr-env-orphan"),
    )
    monkeypatch.setattr(instance, "_managed_secret_gc_now", lambda: GC_NOW)
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name, _candidate: deleted.append(name) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == ["alpha-vsr-env-orphan"]


def test_gc_is_fail_closed_when_history_cannot_be_verified(monkeypatch):
    instance = k8s_backend()
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", lambda: None)
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_candidates",
        lambda: ["alpha-vsr-env-orphan"],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name, _candidate: deleted.append(name),
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert deleted == []


def test_gc_is_fail_closed_when_kubectl_refs_cannot_be_verified(monkeypatch):
    instance = k8s_backend()
    events = []
    monkeypatch.setattr(
        instance,
        "_current_router_secret_refs",
        lambda: (_ for _ in ()).throw(RuntimeError("kubectl failed")),
    )
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_candidates",
        lambda: events.append("list") or ["alpha-vsr-env-orphan"],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name, _candidate: events.append(("delete", name)) or True,
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert events == []


def test_gc_retains_fresh_unreferenced_secret_from_concurrent_deploy(monkeypatch):
    instance = k8s_backend()
    name = "alpha-vsr-env-deploy-a-in-flight"
    payload = managed_secret_created_at(
        name,
        kubernetes_timestamp(GC_NOW - timedelta(minutes=14)),
    )
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", set)
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(instance, "_managed_secret_gc_now", lambda: GC_NOW)
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda value, _candidate: deleted.append(value) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == []


def test_gc_deletes_unreferenced_secret_only_after_grace(monkeypatch):
    instance = k8s_backend()
    name = "alpha-vsr-env-stale"
    payload = managed_secret_created_at(
        name,
        kubernetes_timestamp(STALE_CREATED_AT),
    )
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", set)
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(instance, "_managed_secret_gc_now", lambda: GC_NOW)
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda value, _candidate: deleted.append(value) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == [name]


def test_gc_retains_old_secret_until_teardown_quarantine_expires(monkeypatch):
    instance = k8s_backend()
    name = "alpha-vsr-env-teardown-quarantine"
    payload = annotate_delete_not_before(
        managed_secret_created_at(name, kubernetes_timestamp(STALE_CREATED_AT)),
        kubernetes_timestamp(GC_NOW + timedelta(minutes=1)),
    )
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", set)
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(instance, "_managed_secret_gc_now", lambda: GC_NOW)
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda value, _candidate: deleted.append(value) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == []


@pytest.mark.parametrize(
    "created_at",
    [
        None,
        "not-an-rfc3339-timestamp",
        "2026-07-13T11:00:00",
        "2026-07-13T11:00:00+00:60",
        "2026-07-13T11:00:00+24:00",
        "2026-07-13T11:00:00-00:00",
        "2026-07-13T11:00:00.1234567890Z",
    ],
)
def test_gc_fails_closed_on_malformed_creation_timestamp(monkeypatch, created_at):
    instance = k8s_backend()
    name = "alpha-vsr-env-unknown-age"
    payload = managed_secret_created_at(name, created_at)
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", set)
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda value, _candidate: deleted.append(value) or True,
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert deleted == []


@pytest.mark.parametrize(
    "delete_not_before",
    [
        "not-an-rfc3339-timestamp",
        "2026-07-13T11:00:00+00:60",
        "2026-07-13T11:00:00-00:00",
    ],
)
def test_gc_fails_closed_on_malformed_quarantine_timestamp(
    monkeypatch, delete_not_before
):
    instance = k8s_backend()
    name = "alpha-vsr-env-unknown-quarantine"
    payload = annotate_delete_not_before(
        managed_secret_created_at(name, kubernetes_timestamp(STALE_CREATED_AT)),
        delete_not_before,
    )
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", set)
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name, _candidate: deleted.append(name),
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert deleted == []


def test_gc_rechecks_refs_after_grace_before_deleting(monkeypatch):
    instance = k8s_backend()
    name = "alpha-vsr-env-became-referenced"
    payload = managed_secret_created_at(
        name,
        kubernetes_timestamp(STALE_CREATED_AT),
    )
    protected_snapshots = iter([set(), {name}])
    monkeypatch.setattr(
        instance,
        "_protected_secret_refs",
        lambda: next(protected_snapshots),
    )
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(instance, "_managed_secret_gc_now", lambda: GC_NOW)
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda _name, _candidate: pytest.fail(
            "a newly referenced Secret must be retained"
        ),
    )

    assert instance._garbage_collect_managed_secrets() is True


def test_helm_history_manifests_form_the_rollback_protection_set(monkeypatch):
    instance = k8s_backend()
    monkeypatch.setattr("cli.k8s_backend.HELM_HISTORY_MAX", 3)
    calls = []
    responses = iter(
        [
            MagicMock(
                returncode=0,
                stdout=json.dumps([{"revision": 8}, {"revision": 9}]),
            ),
            MagicMock(returncode=0, stdout=deployment_manifest("old-secret")),
            MagicMock(returncode=0, stdout=deployment_manifest("new-secret")),
        ]
    )

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return next(responses)

    monkeypatch.setattr("subprocess.run", run)

    refs = instance._helm_retained_secret_refs()

    assert {"old-secret", "new-secret"} <= refs
    history_cmd = calls[0][0]
    assert history_cmd[history_cmd.index("--max") + 1] == "3"
    assert [call[0][call[0].index("--revision") + 1] for call in calls[1:]] == [
        "8",
        "9",
    ]
    assert all(call[1]["capture_output"] is True for call in calls)


@pytest.mark.parametrize(
    "history_result",
    [
        MagicMock(returncode=1, stdout="", stderr="sensitive-server-output"),
        MagicMock(returncode=0, stdout="not-json", stderr=""),
    ],
)
def test_helm_history_failure_retains_secrets_without_logging_output(
    monkeypatch, caplog, history_result
):
    instance = k8s_backend()
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: history_result)

    assert instance._helm_retained_secret_refs() is None
    assert "sensitive-server-output" not in caplog.text


def test_delete_uses_uid_and_resource_version_preconditions(monkeypatch):
    instance = k8s_backend()
    name = "alpha-vsr-env-orphan"
    candidate = managed_candidates(name)[name]
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return MagicMock(returncode=0)

    monkeypatch.setattr(instance, "_run", run)

    assert instance._delete_secret_if_exists(name, candidate) is True
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "kubectl",
        "delete",
        "--raw",
        f"/api/v1/namespaces/shared-ns/secrets/{name}",
        "-f",
        "-",
    ]
    delete_options = json.loads(kwargs["input_text"])
    assert delete_options["kind"] == "DeleteOptions"
    assert delete_options["preconditions"] == {
        "uid": candidate.uid,
        "resourceVersion": candidate.resource_version,
    }
    assert kwargs["check"] is False


@pytest.mark.parametrize("missing_field", ["uid", "resourceVersion"])
def test_gc_fails_closed_without_delete_precondition_identity(
    monkeypatch, missing_field
):
    instance = k8s_backend()
    payload = managed_secret_created_at(
        "alpha-vsr-env-unsafe-identity",
        kubernetes_timestamp(STALE_CREATED_AT),
    )
    payload["metadata"].pop(missing_field)
    monkeypatch.setattr(instance, "_protected_secret_refs", set)
    monkeypatch.setattr(
        instance, "_get_json", lambda *args, **kwargs: {"items": [payload]}
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda *_args: pytest.fail("Secret without CAS identity must be retained"),
    )

    assert instance._garbage_collect_managed_secrets() is False


def test_delete_failure_is_reported_for_later_idempotent_retry(monkeypatch, caplog):
    instance = k8s_backend()
    name = "alpha-vsr-env-orphan"
    monkeypatch.setattr(
        instance,
        "_run",
        lambda *args, **kwargs: MagicMock(returncode=1),
    )

    assert (
        instance._delete_secret_if_exists(name, managed_candidates(name)[name]) is False
    )
    assert "later cleanup will retry" in caplog.text


def test_legacy_secret_name_remains_a_module_compatibility_seam(monkeypatch):
    instance = k8s_backend()
    monkeypatch.setattr("cli.k8s_backend.ENV_SECRET_NAME", "legacy-test-secret")
    monkeypatch.setattr(
        instance,
        "_run",
        lambda *args, **kwargs: pytest.fail("the legacy Secret must be retained"),
    )

    assert instance._delete_secret_if_exists("legacy-test-secret") is True
