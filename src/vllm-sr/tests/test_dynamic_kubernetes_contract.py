from __future__ import annotations

import pytest
import yaml
from cli.config_translator import translate_config_to_helm_values


def _durable_config(*, access_enabled=False):
    return {
        "version": "v0.3",
        "listeners": [{"name": "public", "address": "0.0.0.0", "port": 8899}],
        "providers": {"models": []},
        "routing": {"modelCards": []},
        "recipes": [],
        "entrypoints": [],
        "global": {
            "stores": {
                "management": {"postgres": {"dsn_env": "ACCESS_DATABASE_URL"}},
                "runtime": {"redis": {"url_env": "ACCESS_RUNTIME_URL"}},
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
                    "enabled": True,
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
                    "secretKeyRef": {"name": "router-stores", "key": "redis"}
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


def test_kubernetes_preserves_one_atomic_v03_config(tmp_path):
    config = _durable_config()

    values = _translate(tmp_path, config)

    assert values["configOverride"] == config
    assert "control_plane" not in values["configOverride"]["global"]
    assert values["extraEnv"] == _secret_profile()["extraEnv"]


def test_kubernetes_requires_every_external_store_secret(tmp_path):
    profile = _secret_profile()
    profile["extraEnv"] = profile["extraEnv"][:1]

    with pytest.raises(ValueError, match="Valkey reference"):
        _translate(tmp_path, _durable_config(), profile=profile)


def test_access_capability_is_derived_without_public_mode(tmp_path):
    config = _durable_config(access_enabled=True)

    values = _translate(tmp_path, config)

    assert values["configOverride"]["global"]["services"]["access"]["enabled"] is True


def test_store_file_references_must_be_read_only_secret_mounts(tmp_path):
    config = _durable_config()
    config["global"]["stores"]["management"]["postgres"] = {
        "dsn_file": "/run/router-secrets/postgres"
    }
    config["global"]["stores"]["runtime"]["redis"] = {
        "url_file": "/run/router-secrets/redis"
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


def test_file_only_config_needs_no_store_secret(tmp_path):
    config = {
        "version": "v0.3",
        "providers": {"models": []},
        "routing": {"modelCards": []},
    }

    values = _translate(tmp_path, config, profile={})

    assert values["configOverride"] == config
