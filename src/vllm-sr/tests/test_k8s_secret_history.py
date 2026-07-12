"""Credential and rollback-safety tests for CLI-managed Kubernetes Secrets."""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.k8s_backend import HELM_HISTORY_MAX, K8sBackend  # noqa: E402
from cli.k8s_secret_history import (  # noqa: E402
    deployment_secret_refs,
    parse_helm_history_revisions,
    parse_helm_manifest_secret_refs,
)
from cli.k8s_secret_plan import (  # noqa: E402
    SecretSnapshot,
    desired_encoded_secret_data,
    encode_secret_data,
)


def backend() -> K8sBackend:
    return K8sBackend(
        namespace="shared-ns",
        context=None,
        release_name="alpha",
        chart_dir="/chart",
    )


def deployment_manifest(secret_name: str) -> str:
    return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: router
spec:
  template:
    spec:
      imagePullSecrets:
      - name: image-pull-secret
      initContainers:
      - name: init
        env:
        - name: TOKEN
          valueFrom:
            secretKeyRef:
              name: init-secret
              key: token
      containers:
      - name: router
        envFrom:
        - secretRef:
            name: {secret_name}
      volumes:
      - name: direct
        secret:
          secretName: volume-secret
      - name: projected
        projected:
          sources:
          - secret:
              name: projected-secret
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: ignored
data:
  note: secretRef name should-not-be-parsed
"""


def deployment_with_volumes(
    volumes: list[dict],
    *,
    env_secret: str | None = None,
) -> dict:
    container = {"name": "router", "image": "example.invalid/router:test"}
    if env_secret is not None:
        container["envFrom"] = [{"secretRef": {"name": env_secret}}]
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "router"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [container],
                    "volumes": volumes,
                }
            }
        },
    }


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


def test_manifest_parser_uses_safe_yaml_and_only_deployment_pod_specs():
    assert parse_helm_manifest_secret_refs(deployment_manifest("runtime-secret")) == {
        "image-pull-secret",
        "init-secret",
        "projected-secret",
        "runtime-secret",
        "volume-secret",
    }
    assert parse_helm_manifest_secret_refs("kind: [") is None
    assert (
        parse_helm_manifest_secret_refs("kind: ConfigMap\nmetadata: {name: only}")
        is None
    )


@pytest.mark.parametrize(
    "manifest",
    [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata: {name: router}\nspec: truncated\n",
        "apiVersion: apps/v1\nkind: Deployment\nspec:\n  template:\n    spec:\n      containers: invalid\n",
    ],
)
def test_manifest_parser_fails_closed_on_structurally_malformed_deployment(manifest):
    assert parse_helm_manifest_secret_refs(manifest) is None


@pytest.mark.parametrize(
    ("source_name", "source"),
    [
        ("secret", {"secretName": "protected-secret"}),
        (
            "projected",
            {"sources": [{"secret": {"name": "protected-secret"}}]},
        ),
        (
            "csi",
            {
                "driver": "example.csi.invalid",
                "nodePublishSecretRef": {"name": "protected-secret"},
            },
        ),
        (
            "azureFile",
            {"secretName": "protected-secret", "shareName": "data"},
        ),
        (
            "cephfs",
            {
                "monitors": ["ceph.example.invalid"],
                "secretRef": {"name": "protected-secret"},
            },
        ),
        (
            "cinder",
            {"volumeID": "volume-id", "secretRef": {"name": "protected-secret"}},
        ),
        (
            "flexVolume",
            {"driver": "example/driver", "secretRef": {"name": "protected-secret"}},
        ),
        (
            "iscsi",
            {
                "targetPortal": "127.0.0.1:3260",
                "iqn": "iqn.example",
                "lun": 0,
                "secretRef": {"name": "protected-secret"},
            },
        ),
        (
            "rbd",
            {
                "monitors": ["rbd.example.invalid"],
                "image": "image",
                "secretRef": {"name": "protected-secret"},
            },
        ),
        (
            "scaleIO",
            {
                "gateway": "scaleio.example.invalid",
                "system": "system",
                "secretRef": {"name": "protected-secret"},
            },
        ),
        (
            "storageos",
            {"secretRef": {"name": "protected-secret"}},
        ),
    ],
)
def test_volume_parser_covers_every_secret_bearing_pod_volume(
    source_name,
    source,
):
    deployment = deployment_with_volumes(
        [{"name": "credential-volume", source_name: source}]
    )

    assert deployment_secret_refs(deployment) == {"protected-secret"}


def test_volume_parser_accepts_normal_non_secret_sources():
    deployment = deployment_with_volumes(
        [
            {"name": "scratch", "emptyDir": {}},
            {"name": "settings", "configMap": {"name": "settings"}},
            {"name": "host", "hostPath": {"path": "/tmp"}},
            {"name": "claim", "persistentVolumeClaim": {"claimName": "data"}},
            {"name": "network", "nfs": {"server": "127.0.0.1", "path": "/"}},
            {
                "name": "artifact",
                "image": {"reference": "example.invalid/artifact:test"},
            },
            {
                "name": "public-csi",
                "csi": {"driver": "example.csi.invalid"},
            },
            {
                "name": "public-flex",
                "flexVolume": {"driver": "example/driver"},
            },
        ]
    )

    assert deployment_secret_refs(deployment) == set()


@pytest.mark.parametrize(
    "volume",
    [
        {"name": "future", "futureStorage": {"secretRef": {"name": "live"}}},
        {"name": "missing-source"},
        {"name": "multiple", "emptyDir": {}, "secret": {"secretName": "live"}},
        {
            "name": "bad-csi",
            "csi": {
                "driver": "example.csi.invalid",
                "nodePublishSecretRef": "live",
            },
        },
        {"name": "bad-scaleio", "scaleIO": {"system": "system"}},
        {"name": "bad-azure-file", "azureFile": {"shareName": "data"}},
        {"name": "bad-shape", "emptyDir": []},
        {
            "name": "future-projection",
            "projected": {"sources": [{"futureProjection": {"name": "live"}}]},
        },
    ],
)
def test_volume_parser_fails_closed_on_unknown_or_incomplete_sources(volume):
    manifest = yaml.safe_dump(deployment_with_volumes([volume]))

    assert parse_helm_manifest_secret_refs(manifest) is None


def test_unknown_volume_source_prevents_gc_deletion(monkeypatch):
    instance = backend()
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
        "_list_release_managed_secret_names",
        lambda: events.append("list") or ["alpha-vsr-env-live"],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name: events.append(("delete", name)) or True,
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert events == []


def test_csi_reference_protects_current_and_rollback_generations(monkeypatch):
    instance = backend()
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
        "_list_release_managed_secret_names",
        lambda: [active, rollback, orphan],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name: deleted.append(name) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == [orphan]


def test_first_and_unchanged_shared_looper_revisions_persist_recreate(monkeypatch):
    instance = backend()
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [])
    first = instance._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64})
    assert first.recreate_for_looper_rotation is True

    current_name = first.active_name
    manifest = json.loads(first.new_manifest)
    payload = {
        "metadata": {
            "labels": {
                "app.kubernetes.io/managed-by": "vllm-sr-cli",
                "app.kubernetes.io/instance": "alpha",
                "semantic-router.vllm.ai/runtime-env-secret": "true",
            }
        },
        "immutable": True,
        "data": manifest["data"],
    }
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [current_name])
    monkeypatch.setattr(instance, "_get_secret_json", lambda _: payload)

    unchanged = instance._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64})
    assert unchanged.active_name == current_name
    assert unchanged.creates_secret is False
    assert unchanged.recreate_for_looper_rotation is True


def test_explicit_looper_rotation_creates_exact_new_generation(monkeypatch):
    instance = backend()
    current_name = "alpha-vsr-env-current"
    payload = json.loads(instance._secret_manifest(current_name, {}))
    payload["data"] = instance._encode_secret_data(
        {"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64, "HF_TOKEN": "omitted"}
    )
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [current_name])
    monkeypatch.setattr(instance, "_get_secret_json", lambda _: payload)

    rotated = instance._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": "b" * 64})
    assert rotated.active_name != current_name
    assert json.loads(rotated.new_manifest)["data"] == instance._encode_secret_data(
        {"VLLM_SR_LOOPER_SHARED_SECRET": "b" * 64}
    )
    assert rotated.recreate_for_looper_rotation is True


@pytest.mark.parametrize(
    ("cli_managed", "previous_value"),
    [
        (False, encode_secret_data({"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64})),
        (True, {"VLLM_SR_LOOPER_SHARED_SECRET": "not-valid-base64"}),
    ],
)
def test_unmanaged_or_damaged_previous_looper_key_is_not_reused(
    monkeypatch, cli_managed, previous_value
):
    generated = "c" * 64
    previous = SecretSnapshot(
        name="alpha-vsr-env-previous",
        data=previous_value,
        immutable=True,
        cli_managed=cli_managed,
        data_known=True,
    )
    monkeypatch.setattr("cli.k8s_secret_plan.secrets.token_hex", lambda _: generated)

    desired = desired_encoded_secret_data(None, previous)
    encoded = desired["VLLM_SR_LOOPER_SHARED_SECRET"]
    assert base64.b64decode(encoded).decode() == generated


def test_gc_retains_current_and_every_rollback_revision(monkeypatch):
    instance = backend()
    retained = {"alpha-vsr-env-current", "alpha-vsr-env-rollback"}
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", lambda: retained)
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_names",
        lambda: [*sorted(retained), "alpha-vsr-env-orphan"],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name: deleted.append(name) or True,
    )

    assert instance._garbage_collect_managed_secrets() is True
    assert deleted == ["alpha-vsr-env-orphan"]


def test_gc_is_fail_closed_when_history_cannot_be_verified(monkeypatch):
    instance = backend()
    deleted = []
    monkeypatch.setattr(instance, "_protected_secret_refs", lambda: None)
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_names",
        lambda: ["alpha-vsr-env-orphan"],
    )
    monkeypatch.setattr(instance, "_delete_secret_if_exists", deleted.append)

    assert instance._garbage_collect_managed_secrets() is False
    assert deleted == []


def test_gc_is_fail_closed_when_kubectl_refs_cannot_be_verified(monkeypatch):
    instance = backend()
    events = []
    monkeypatch.setattr(
        instance,
        "_current_router_secret_refs",
        lambda: (_ for _ in ()).throw(RuntimeError("kubectl failed")),
    )
    monkeypatch.setattr(
        instance,
        "_list_release_managed_secret_names",
        lambda: events.append("list") or ["alpha-vsr-env-orphan"],
    )
    monkeypatch.setattr(
        instance,
        "_delete_secret_if_exists",
        lambda name: events.append(("delete", name)) or True,
    )

    assert instance._garbage_collect_managed_secrets() is False
    assert events == []


def test_helm_history_manifests_form_the_rollback_protection_set(monkeypatch):
    instance = backend()
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
    assert history_cmd[history_cmd.index("--max") + 1] == str(HELM_HISTORY_MAX)
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
    instance = backend()
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: history_result)

    assert instance._helm_retained_secret_refs() is None
    assert "sensitive-server-output" not in caplog.text


def test_delete_failure_is_reported_for_later_idempotent_retry(monkeypatch, caplog):
    instance = backend()
    monkeypatch.setattr(
        instance,
        "_run",
        lambda *args, **kwargs: MagicMock(returncode=1),
    )

    assert instance._delete_secret_if_exists("alpha-vsr-env-orphan") is False
    assert "later cleanup will retry" in caplog.text
