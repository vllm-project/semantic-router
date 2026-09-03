"""Kubernetes credential Secret planning and Helm binding tests."""

from __future__ import annotations

import copy
import json

import pytest
import yaml
from cli.config_translator import translate_config_to_helm_values
from cli.k8s_env_secret import (
    ENV_SECRET_MANAGER_LABEL,
    ENV_SECRET_MANAGER_VALUE,
    ENV_SECRET_OWNER_LABEL,
    ENV_SECRET_REVISION_ANNOTATION,
    LEGACY_ENV_SECRET_NAME,
    EnvSecretPlan,
    build_env_secret_plan,
    env_secret_owner,
    is_managed_env_secret_name,
    managed_env_secret_names,
    referenced_secret_names,
)
from cli.recipe_topology_contract import MANAGEMENT_CREDENTIAL_ENV
from k8s_env_secret_test_support import _backend, _plan


def test_plan_is_release_scoped_revisioned_and_secret_stays_in_manifest(
    monkeypatch, caplog
):
    monkeypatch.setattr("cli.k8s_env_secret.secrets.token_hex", lambda _size: "a" * 32)
    canary = "management-secret-canary"

    with caplog.at_level("DEBUG"):
        plan = build_env_secret_plan(
            namespace="team-a",
            release_name="router-a",
            env_vars={"CUSTOM_TOKEN": canary, "PUBLIC_ENDPOINT": "https://example"},
            sensitive_names={"CUSTOM_TOKEN"},
        )

    assert plan is not None
    assert len(plan.name) <= 63
    assert is_managed_env_secret_name(plan.name, plan.owner)
    manifest = json.loads(plan.manifest)
    assert manifest["stringData"] == {"CUSTOM_TOKEN": canary}
    assert manifest["immutable"] is True
    assert manifest["metadata"]["labels"] == {
        ENV_SECRET_MANAGER_LABEL: ENV_SECRET_MANAGER_VALUE,
        ENV_SECRET_OWNER_LABEL: plan.owner,
    }
    assert canary not in caplog.text


def test_plan_revisions_and_release_owners_do_not_collide(monkeypatch):
    revisions = iter(("a" * 32, "b" * 32, "c" * 32))
    monkeypatch.setattr(
        "cli.k8s_env_secret.secrets.token_hex", lambda _size: next(revisions)
    )

    first = build_env_secret_plan(
        namespace="team-a",
        release_name="router-a",
        env_vars={"HF_TOKEN": "one"},
        sensitive_names={"HF_TOKEN"},
    )
    second = build_env_secret_plan(
        namespace="team-a",
        release_name="router-a",
        env_vars={"HF_TOKEN": "two"},
        sensitive_names={"HF_TOKEN"},
    )
    other_release = build_env_secret_plan(
        namespace="team-a",
        release_name="router-b",
        env_vars={"HF_TOKEN": "three"},
        sensitive_names={"HF_TOKEN"},
    )

    assert first is not None and second is not None and other_release is not None
    assert first.name != second.name
    assert first.owner == second.owner
    assert first.owner != other_release.owner


def test_plan_omits_empty_or_non_sensitive_values_and_rejects_non_string():
    assert (
        build_env_secret_plan(
            namespace="team-a",
            release_name="router-a",
            env_vars={"HF_TOKEN": "", "HF_ENDPOINT": "https://example"},
            sensitive_names={"HF_TOKEN"},
        )
        is None
    )
    with pytest.raises(ValueError, match="must be a string"):
        build_env_secret_plan(
            namespace="team-a",
            release_name="router-a",
            env_vars={"HF_TOKEN": 7},  # type: ignore[dict-item]
            sensitive_names={"HF_TOKEN"},
        )


def test_metadata_only_name_parsers_filter_invalid_or_other_owner_names():
    owner = env_secret_owner("team-a", "router-a")
    valid = f"vllm-sr-env-{owner}-{'a' * 32}"
    other = f"vllm-sr-env-{'b' * 16}-{'c' * 32}"

    assert referenced_secret_names(f"{valid}\noperator-db\n") == {
        valid,
        "operator-db",
    }
    assert managed_env_secret_names(f"{valid}\n{other}\nnot-a-secret\n", owner) == {
        valid
    }


def test_profile_env_and_secret_refs_are_preserved_without_mutation(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"listeners": []}), encoding="utf-8")
    profile = {
        "env": [{"name": "PROFILE_ENV", "value": "profile"}],
        "envFromSecrets": ["operator-db-secret"],
        "podAnnotations": {"operator.example/owned": "true"},
        "configOverride": {"providers": {"models": [{"name": "stale-profile"}]}},
    }
    original = copy.deepcopy(profile)

    values = translate_config_to_helm_values(
        str(config),
        profile_values=profile,
        env_vars={"PROFILE_ENV": "override", "HF_ENDPOINT": "https://hf"},
        env_secret_name="managed-secret",
    )

    assert values["env"] == [
        {"name": "PROFILE_ENV", "value": "profile"},
        {"name": "HF_ENDPOINT", "value": "https://hf"},
    ]
    assert values["envFromSecrets"] == ["operator-db-secret", "managed-secret"]
    assert values["podAnnotations"] == {"operator.example/owned": "true"}
    assert values["configOverride"] == {"listeners": []}
    assert profile == original


@pytest.mark.parametrize("env_field", ["env", "extraEnv"])
def test_sensitive_profile_env_requires_external_value_from(tmp_path, env_field):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "providers": {"models": [{"api_key": "${GEMINI_API_KEY}"}]},
            }
        ),
        encoding="utf-8",
    )
    canary = "profile-literal-secret-canary"
    profile = {env_field: [{"name": "GEMINI_API_KEY", "value": canary}]}

    with pytest.raises(ValueError, match="GEMINI_API_KEY") as exc:
        translate_config_to_helm_values(
            str(config),
            profile_values=profile,
            env_vars={"GEMINI_API_KEY": "host-secret-canary"},
            env_secret_name="managed-secret",
        )

    assert canary not in str(exc.value)

    external = {
        "name": "GEMINI_API_KEY",
        "valueFrom": {
            "secretKeyRef": {"name": "operator-model-secret", "key": "api-key"}
        },
    }
    values = translate_config_to_helm_values(
        str(config),
        profile_values={env_field: [external]},
        env_vars={"GEMINI_API_KEY": "host-secret-canary"},
        env_secret_name="managed-secret",
    )
    assert values[env_field] == [external]


def test_profile_configmap_credentials_require_environment_and_secret_ref(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"listeners": []}), encoding="utf-8")
    canary = "profile-configmap-secret-canary"
    literal_profile = {
        "dependencies": {"semanticCache": {"redis": {"password": canary}}}
    }

    with pytest.raises(
        ValueError, match=r"dependencies\.semanticCache\.redis\.password"
    ) as exc:
        translate_config_to_helm_values(str(config), profile_values=literal_profile)
    assert canary not in str(exc.value)

    with pytest.raises(ValueError, match="must bind REDIS_PASSWORD"):
        translate_config_to_helm_values(
            str(config),
            profile_values={
                "dependencies": {
                    "semanticCache": {"redis": {"password": "${REDIS_PASSWORD}"}}
                }
            },
            env_vars={"REDIS_PASSWORD": "host-only-secret-canary"},
            env_secret_name="managed-source-secret",
        )

    with pytest.raises(ValueError, match="uppercase, non-reserved"):
        translate_config_to_helm_values(
            str(config),
            profile_values={
                "dependencies": {"semanticCache": {"redis": {"password": "${HOME}"}}},
                "extraEnv": [
                    {
                        "name": "HOME",
                        "valueFrom": {
                            "secretKeyRef": {
                                "name": "operator-redis",
                                "key": "password",
                            }
                        },
                    }
                ],
            },
        )

    external = {
        "name": "REDIS_PASSWORD",
        "valueFrom": {"secretKeyRef": {"name": "operator-redis", "key": "password"}},
    }
    values = translate_config_to_helm_values(
        str(config),
        profile_values={
            "dependencies": {
                "semanticCache": {"redis": {"password": "${REDIS_PASSWORD}"}}
            },
            "extraEnv": [external],
        },
    )
    assert values["extraEnv"] == [external]


def test_profile_kubernetes_secret_references_are_not_literal_credentials(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"listeners": []}), encoding="utf-8")
    profile = {
        "dashboard": {
            "jwtSecret": {
                "existingSecret": "operator-dashboard-jwt",
                "existingSecretKey": "jwt-secret",
            }
        }
    }

    values = translate_config_to_helm_values(
        str(config),
        profile_values=profile,
    )

    assert values["dashboard"]["jwtSecret"] == profile["dashboard"]["jwtSecret"]


def test_atomic_config_drops_ignored_profile_config_from_helm_values(tmp_path):
    config = tmp_path / "config.yaml"
    source = {"version": "v0.3", "listeners": []}
    config.write_text(yaml.safe_dump(source), encoding="utf-8")
    canary = "stale-profile-secret-canary"

    values = translate_config_to_helm_values(
        str(config),
        profile_values={
            "config": {"providers": {"models": [{"api_key": canary}]}},
            "configOverride": {"api_key": canary},
        },
    )

    assert "config" not in values
    assert values["configOverride"] == source
    assert canary not in yaml.safe_dump(values)


def test_sensitive_profile_env_rejects_configmap_value_from(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"version": "v0.3", "api_key": "${MODEL_API_KEY}"}),
        encoding="utf-8",
    )
    profile = {
        "extraEnv": [
            {
                "name": "MODEL_API_KEY",
                "valueFrom": {"configMapKeyRef": {"name": "unsafe", "key": "api-key"}},
            }
        ]
    }

    with pytest.raises(ValueError, match="Secret secretKeyRef"):
        translate_config_to_helm_values(str(config), profile_values=profile)


def test_modern_router_config_and_management_auth_survive_helm_translation(tmp_path):
    config = tmp_path / "config.yaml"
    source = {
        "version": "v0.3",
        "listeners": [{"name": "http", "port": 8899}],
        "providers": {"models": [{"name": "local/custom"}]},
        "routing": {"decisions": [{"name": "custom-route"}]},
        "recipes": [{"name": "custom-recipe"}],
        "global": {
            "services": {
                "management_api": {
                    "remote_exposure": True,
                    "auth": {
                        "mode": "bearer",
                        "tokens": [{"env": "CUSTOM_MGMT_TOKEN", "role": "viewer"}],
                    },
                }
            }
        },
    }
    config.write_text(yaml.safe_dump(source), encoding="utf-8")
    stale_effective = tmp_path / "stale-effective.yaml"
    stale_effective.write_text(
        yaml.safe_dump({"version": "v0.3", "providers": {"models": []}}),
        encoding="utf-8",
    )

    values = translate_config_to_helm_values(
        str(stale_effective),
        config_document=source,
        source_config_file=str(config),
        env_vars={"CUSTOM_MGMT_TOKEN": "secret-value-canary"},
        env_secret_name="managed-secret",
    )

    assert values["configOverride"] == source
    assert "config" not in values
    assert values["envFromSecrets"] == ["managed-secret"]
    assert "secret-value-canary" not in yaml.safe_dump(values)


def test_revision_binding_preserves_profile_values_and_controls_reserved_keys():
    backend = _backend()
    active = _plan(backend).name
    old = _plan(backend, "b" * 32).name
    values = {
        "envFromSecrets": ["operator-db-secret", LEGACY_ENV_SECRET_NAME, old],
        "podAnnotations": {
            "operator.example/owned": "true",
            ENV_SECRET_REVISION_ANNOTATION: "profile-must-not-win",
        },
    }

    backend._bind_env_secret_revision(values, active)

    assert values["envFromSecrets"] == ["operator-db-secret", active]
    assert values["podAnnotations"] == {
        "operator.example/owned": "true",
        ENV_SECRET_REVISION_ANNOTATION: active,
    }
    backend._bind_env_secret_revision(values, None)
    assert values["envFromSecrets"] == ["operator-db-secret"]
    assert values["podAnnotations"] == {"operator.example/owned": "true"}


def test_dashboard_receives_only_the_management_credential_key():
    backend = _backend()
    plan = EnvSecretPlan(
        owner=env_secret_owner(backend.namespace, backend.release_name),
        name=_plan(backend).name,
        manifest="{}",
        key_count=2,
        keys=frozenset({MANAGEMENT_CREDENTIAL_ENV, "HF_TOKEN"}),
    )
    values = {
        "dashboard": {
            "enabled": True,
            "extraEnv": [{"name": "DASHBOARD_PUBLIC", "value": "true"}],
        }
    }

    backend._bind_dashboard_management_credential(values, plan)

    assert values["dashboard"]["extraEnv"] == [
        {"name": "DASHBOARD_PUBLIC", "value": "true"},
        {
            "name": MANAGEMENT_CREDENTIAL_ENV,
            "valueFrom": {
                "secretKeyRef": {
                    "name": plan.name,
                    "key": MANAGEMENT_CREDENTIAL_ENV,
                }
            },
        },
    ]
    assert "HF_TOKEN" not in yaml.safe_dump(values)


def test_dashboard_management_key_covers_custom_chart_default_without_profile():
    backend = _backend()
    plan = EnvSecretPlan(
        owner=env_secret_owner(backend.namespace, backend.release_name),
        name=_plan(backend).name,
        manifest="{}",
        key_count=1,
        keys=frozenset({MANAGEMENT_CREDENTIAL_ENV}),
    )
    values = {}

    backend._bind_dashboard_management_credential(values, plan)

    assert values["dashboard"]["extraEnv"][0]["valueFrom"]["secretKeyRef"] == {
        "name": plan.name,
        "key": MANAGEMENT_CREDENTIAL_ENV,
    }

    minimal_values = {"dashboard": {"enabled": False}}
    backend._bind_dashboard_management_credential(minimal_values, plan)
    assert "extraEnv" not in minimal_values["dashboard"]


def test_dashboard_management_binding_preserves_external_reference_and_removes_stale():
    backend = _backend()
    stale = _plan(backend).name
    values = {
        "dashboard": {
            "enabled": True,
            "extraEnv": [
                {
                    "name": MANAGEMENT_CREDENTIAL_ENV,
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": stale,
                            "key": MANAGEMENT_CREDENTIAL_ENV,
                        }
                    },
                }
            ],
        }
    }
    backend._bind_dashboard_management_credential(values, None)
    assert "extraEnv" not in values["dashboard"]

    external = {
        "name": MANAGEMENT_CREDENTIAL_ENV,
        "valueFrom": {
            "secretKeyRef": {
                "name": "operator-management-secret",
                "key": "token",
            }
        },
    }
    values = {"dashboard": {"enabled": True, "extraEnv": [external]}}
    backend._bind_dashboard_management_credential(values, None)
    assert values["dashboard"]["extraEnv"] == [external]


@pytest.mark.parametrize(
    "entry",
    [
        {"name": MANAGEMENT_CREDENTIAL_ENV, "value": "literal-canary"},
        {
            "name": MANAGEMENT_CREDENTIAL_ENV,
            "valueFrom": {"configMapKeyRef": {"name": "unsafe", "key": "token"}},
        },
    ],
)
def test_dashboard_management_credential_rejects_plain_or_non_secret_refs(entry):
    backend = _backend()
    values = {"dashboard": {"enabled": True, "extraEnv": [entry]}}

    with pytest.raises(ValueError, match="management credential") as exc:
        backend._bind_dashboard_management_credential(values, None)

    assert "literal-canary" not in str(exc.value)
