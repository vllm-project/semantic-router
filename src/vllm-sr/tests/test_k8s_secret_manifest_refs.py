"""Fail-closed manifest parsing tests for Kubernetes Secret references."""

import sys
from pathlib import Path

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
from cli.k8s_secret_history import (  # noqa: E402
    deployment_secret_refs,
    parse_helm_manifest_secret_refs,
)


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
