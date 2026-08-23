from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cli.config_translator import translate_config_to_helm_values


def _managed_config(*, access_enabled=False, include_public_namespace=True):
    control_plane = {"mode": "managed"}
    if include_public_namespace:
        control_plane["public_namespace_id"] = "11111111-1111-4111-8111-111111111111"
    return {
        "version": "v0.4",
        "listeners": [{"name": "public", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "control_plane": control_plane,
            "stores": {
                "access": {
                    "type": "postgres",
                    "postgres": {"dsn_env": "ACCESS_DATABASE_URL"},
                },
                "access_runtime": {
                    "type": "redis",
                    "redis": {"url_env": "ACCESS_RUNTIME_URL"},
                },
            },
            "services": {
                "agent": {
                    "public_inference_endpoint": "http://semantic-router-public.router-system.svc.cluster.local/v1/chat/completions"
                },
                "access": {"enabled": access_enabled},
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
                },
                "backend_dispatch": {
                    "bind_address": "0.0.0.0",
                    "port": 8187,
                    "audience": "vllm-sr.backend-dispatch",
                    "capability_ttl": "30s",
                    "max_request_body_bytes": 67108864,
                },
                "management_api": {
                    "bind_address": "0.0.0.0",
                    "port": 8080,
                    "auth": {"mode": "router"},
                },
            },
        },
    }


def _secret_profile():
    return {
        "extraEnv": [
            {
                "name": "ACCESS_DATABASE_URL",
                "valueFrom": {
                    "secretKeyRef": {"name": "router-stores", "key": "postgres"}
                },
            },
            {
                "name": "ACCESS_RUNTIME_URL",
                "valueFrom": {
                    "secretKeyRef": {"name": "router-stores", "key": "valkey"}
                },
            },
        ]
    }


def _translate(tmp_path, config, *, profile=None):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return translate_config_to_helm_values(
        str(path),
        config_document=config,
        profile_values=profile or _secret_profile(),
    )


def test_managed_kubernetes_preserves_one_atomic_bootstrap_and_dispatch(tmp_path):
    config = _managed_config()

    values = _translate(tmp_path, config)

    assert values["configOverride"] == config
    assert values["configOverride"]["global"]["services"]["backend_dispatch"] == {
        "bind_address": "0.0.0.0",
        "port": 8187,
        "audience": "vllm-sr.backend-dispatch",
        "capability_ttl": "30s",
        "max_request_body_bytes": 67108864,
    }
    assert "providers" not in values["configOverride"]
    assert values["extraEnv"] == _secret_profile()["extraEnv"]


def test_managed_kubernetes_requires_both_external_store_secrets(tmp_path):
    profile = _secret_profile()
    profile["extraEnv"] = profile["extraEnv"][:1]

    with pytest.raises(ValueError, match="Valkey reference"):
        _translate(tmp_path, _managed_config(), profile=profile)


def test_public_namespace_is_only_for_managed_routing_only(tmp_path):
    with pytest.raises(ValueError, match="public_namespace_id"):
        _translate(
            tmp_path,
            _managed_config(include_public_namespace=False),
        )

    access_config = _managed_config(
        access_enabled=True,
        include_public_namespace=False,
    )
    assert _translate(tmp_path, access_config)["configOverride"] == access_config

    with pytest.raises(ValueError, match="public_namespace_id"):
        _translate(tmp_path, _managed_config(access_enabled=True))


def test_managed_file_references_must_be_read_only_secret_mounts(tmp_path):
    config = _managed_config()
    config["global"]["stores"]["access"]["postgres"] = {
        "dsn_file": "/run/router-secrets/postgres"
    }
    config["global"]["stores"]["access_runtime"]["redis"] = {
        "url_file": "/run/router-secrets/valkey"
    }
    profile = {
        "extraVolumes": [
            {"name": "router-stores", "secret": {"secretName": "router-stores"}}
        ],
        "extraVolumeMounts": [
            {
                "name": "router-stores",
                "mountPath": "/run/router-secrets",
                "readOnly": True,
            }
        ],
    }

    values = _translate(tmp_path, config, profile=profile)

    assert values["extraVolumes"] == profile["extraVolumes"]
    assert values["extraVolumeMounts"] == profile["extraVolumeMounts"]


def test_helm_managed_runtime_has_one_shot_migration_and_private_dispatch():
    repository = Path(__file__).resolve().parents[3]
    templates = repository / "deploy/helm/semantic-router/templates"
    migration = (templates / "access-migrate-job.yaml").read_text(encoding="utf-8")
    deployment = (templates / "deployment.yaml").read_text(encoding="utf-8")
    service = (templates / "service.yaml").read_text(encoding="utf-8")

    assert '"helm.sh/hook": pre-install,pre-upgrade' in migration
    assert 'command: ["/usr/local/bin/access-migrate"]' in migration
    assert "kind: Job" in migration
    assert "kind: Deployment" not in migration
    assert "name: backend-dsp" in deployment
    assert "name: backend-dsp" in service
    assert len("backend-dsp") <= 15
    assert "type: ClusterIP" in service


def test_dashboard_e2e_uses_an_atomic_managed_bootstrap():
    repository = Path(__file__).resolve().parents[3]
    values_path = repository / "e2e/profiles/dashboard/values.yaml"
    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))

    assert "config" not in values
    bootstrap = values["configOverride"]
    assert bootstrap["version"] == "v0.4"
    assert bootstrap["global"]["control_plane"]["mode"] == "managed"
    assert "models" not in bootstrap
    assert "recipes" not in bootstrap
    assert "entrypoints" not in bootstrap
