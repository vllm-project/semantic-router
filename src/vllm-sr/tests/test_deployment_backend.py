"""Tests for the deployment backend abstraction and K8s / Docker wiring."""

import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_translator import (  # noqa: E402
    _deep_merge,
    _split_image_reference,
    temporary_helm_values_file,
    translate_config_to_helm_values,
    write_helm_values_file,
)
from cli.container_backend import ContainerBackend  # noqa: E402
from cli.deployment_backend import DEFAULT_TARGET, resolve_target  # noqa: E402
from cli.runtime_lifecycle_lock import RuntimeLifecycleLockError  # noqa: E402

# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_none_returns_default(self):
        assert resolve_target(None) == DEFAULT_TARGET

    def test_docker(self):
        assert resolve_target("docker") == "docker"

    def test_k8s(self):
        assert resolve_target("k8s") == "k8s"

    def test_case_insensitive(self):
        assert resolve_target("K8S") == "k8s"
        assert resolve_target("Docker") == "docker"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid deployment target"):
            resolve_target("aws")


# ---------------------------------------------------------------------------
# ContainerBackend
# ---------------------------------------------------------------------------


class TestContainerBackend:
    def test_deploy_delegates_to_start_vllm_sr(self, monkeypatch):
        captured = {}
        lifecycle_lock = MagicMock()
        monkeypatch.setattr(
            "cli.container_backend.start_vllm_sr",
            lambda *a, **kw: captured.update(kw),
        )
        monkeypatch.setattr(
            "cli.container_backend.acquire_runtime_lifecycle_lock",
            MagicMock(return_value=lifecycle_lock),
        )
        monkeypatch.setattr(
            "cli.container_backend.get_container_runtime", lambda: "docker"
        )

        backend = ContainerBackend()
        runtime_lock = object()
        backend.deploy(
            config_file="/tmp/config.yaml",
            source_config_file="/tmp/source-config.yaml",
            runtime_config_file="/tmp/runtime-config.yaml",
            env_vars={"A": "B"},
            image="test:img",
            router_image="test:router",
            envoy_image="test:envoy",
            dashboard_image="test:dashboard",
            topology="split",
            pull_policy="always",
            enable_observability=False,
            runtime_config_lock=runtime_lock,
        )

        assert captured["source_config_file"] == "/tmp/source-config.yaml"
        assert captured["runtime_config_file"] == "/tmp/runtime-config.yaml"
        assert captured["image"] == "test:img"
        assert captured["router_image"] == "test:router"
        assert captured["envoy_image"] == "test:envoy"
        assert captured["dashboard_image"] == "test:dashboard"
        assert captured["topology"] == "split"
        assert captured["pull_policy"] == "always"
        assert captured["enable_observability"] is False
        assert captured["runtime_config_lock"] is runtime_lock
        lifecycle_lock.__enter__.assert_called_once_with()
        lifecycle_lock.__exit__.assert_called_once()

    def test_teardown_delegates_to_stop_vllm_sr(self, monkeypatch):
        called = []
        lifecycle_lock = MagicMock()
        monkeypatch.setattr(
            "cli.container_backend.stop_vllm_sr",
            lambda: called.append(True),
        )
        acquire_lock = MagicMock(return_value=lifecycle_lock)
        monkeypatch.setattr(
            "cli.container_backend.acquire_runtime_lifecycle_lock", acquire_lock
        )
        monkeypatch.setattr(
            "cli.container_backend.get_container_runtime", lambda: "docker"
        )
        ContainerBackend().teardown()
        assert called
        acquire_lock.assert_called_once_with(runtime="docker", stack_name="vllm-sr")
        lifecycle_lock.__enter__.assert_called_once_with()
        lifecycle_lock.__exit__.assert_called_once()

    @pytest.mark.parametrize("operation", ["deploy", "teardown"])
    def test_mutation_fails_closed_when_lifecycle_lock_is_busy(
        self, monkeypatch, operation
    ):
        start = MagicMock()
        stop = MagicMock()
        monkeypatch.setattr("cli.container_backend.start_vllm_sr", start)
        monkeypatch.setattr("cli.container_backend.stop_vllm_sr", stop)
        monkeypatch.setattr(
            "cli.container_backend.get_container_runtime", lambda: "docker"
        )
        monkeypatch.setattr(
            "cli.container_backend.acquire_runtime_lifecycle_lock",
            MagicMock(
                side_effect=RuntimeLifecycleLockError(
                    "another lifecycle operation in progress"
                )
            ),
        )

        backend = ContainerBackend()
        with pytest.raises(
            RuntimeLifecycleLockError,
            match="another lifecycle operation in progress",
        ):
            if operation == "deploy":
                backend.deploy("config.yaml")
            else:
                backend.teardown()

        start.assert_not_called()
        stop.assert_not_called()

    def test_is_running_false_when_not_found(self, monkeypatch):
        monkeypatch.setattr(
            "cli.container_backend.container_status",
            lambda _: "not found",
        )
        assert ContainerBackend().is_running() is False

    def test_is_running_true_when_split_dashboard_container_running(self, monkeypatch):
        monkeypatch.setattr(
            "cli.container_backend.container_status",
            lambda name: "running" if "dashboard" in name else "not found",
        )
        assert ContainerBackend().is_running() is True

    def test_get_dashboard_url_prefers_split_dashboard_container(self, monkeypatch):
        monkeypatch.setattr(
            "cli.container_backend.container_status",
            lambda name: "running" if "dashboard" in name else "not found",
        )
        assert ContainerBackend().get_dashboard_url() == "http://localhost:8700"


# ---------------------------------------------------------------------------
# Config translator
# ---------------------------------------------------------------------------


class TestConfigTranslator:
    def test_image_override(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": [{"port": 8899}]}))

        values = translate_config_to_helm_values(
            str(config),
            image="myrepo/myimage:v2",
        )
        assert values["image"]["repository"] == "myrepo/myimage"
        assert values["image"]["tag"] == "v2"

    def test_pull_policy_normalised(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config),
            pull_policy="always",
        )
        assert values["image"]["pullPolicy"] == "Always"

    def test_cli_flags_override_profile_deployment_defaults(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}), encoding="utf-8")
        profile = {
            "dashboard": {"enabled": True, "readonly": False},
            "global": {"namespace": "wrong-ns", "imageRegistry": "mirror.invalid"},
        }

        minimal = translate_config_to_helm_values(
            str(config),
            profile_values=profile,
            namespace="cli-ns",
            image="registry.example/router:v2",
            minimal=True,
            readonly=True,
        )
        assert minimal["dashboard"] == {"enabled": False, "readonly": False}
        assert minimal["global"] == {"namespace": "cli-ns", "imageRegistry": ""}

        readonly = translate_config_to_helm_values(
            str(config),
            profile_values=profile,
            namespace="cli-ns",
            readonly=True,
        )
        assert readonly["dashboard"] == {"enabled": True, "readonly": True}
        assert readonly["global"] == {
            "namespace": "cli-ns",
            "imageRegistry": "mirror.invalid",
        }

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            ("repo/router:v2", ("repo/router", "v2")),
            ("localhost:5000/router:v2", ("localhost:5000/router", "v2")),
            ("repo/router", ("repo/router", "latest")),
            ("localhost:5000/router", ("localhost:5000/router", "latest")),
        ],
    )
    def test_image_reference_parsing(self, image, expected):
        assert _split_image_reference(image) == expected

    @pytest.mark.parametrize(
        "image",
        [
            "repo/router@sha256:" + "a" * 64,
            "repo/router:",
            " repo/router:v2",
            "repo/router:v2 ",
        ],
    )
    def test_unsupported_image_reference_fails_closed(self, image):
        with pytest.raises(ValueError, match="Kubernetes"):
            _split_image_reference(image)

    def test_observability_flags(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        enabled = translate_config_to_helm_values(
            str(config), enable_observability=True
        )
        assert enabled["dependencies"]["observability"]["jaeger"]["enabled"] is True

        disabled = translate_config_to_helm_values(
            str(config), enable_observability=False
        )
        assert disabled["dependencies"]["observability"]["jaeger"]["enabled"] is False

    def test_config_sections_pass_through(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "listeners": [{"port": 8899}],
                    "providers": {"models": [{"name": "provider-model"}]},
                    "routing": {"decisions": [{"name": "default"}]},
                    "recipes": [{"name": "custom-recipe"}],
                    "global": {
                        "services": {"management_api": {"auth": {"mode": "bearer"}}}
                    },
                }
            )
        )

        values = translate_config_to_helm_values(str(config))
        override = values["configOverride"]
        assert override["version"] == "v0.3"
        assert override["listeners"] == [{"port": 8899}]
        assert override["providers"]["models"] == [{"name": "provider-model"}]
        assert override["routing"]["decisions"] == [{"name": "default"}]
        assert override["recipes"] == [{"name": "custom-recipe"}]
        assert (
            override["global"]["services"]["management_api"]["auth"]["mode"] == "bearer"
        )

    @pytest.mark.parametrize("config_document", [None, {}, [], "not-a-mapping"])
    def test_empty_or_non_mapping_config_fails_closed(self, tmp_path, config_document):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump(config_document), encoding="utf-8")

        with pytest.raises(ValueError, match="non-empty mapping"):
            translate_config_to_helm_values(str(config))

    @pytest.mark.parametrize(
        ("config_document", "expected_path"),
        [
            ({"access_key": "canary"}, "access_key"),
            ({"api_key": "canary"}, "api_key"),
            ({"auth_token": "canary"}, "auth_token"),
            ({"password": "canary"}, "password"),
            ({"bearer_token": "canary"}, "bearer_token"),
            ({"client-secret": "canary"}, "client-secret"),
            ({"privateKey": "canary"}, "privateKey"),
            ({"api_key": {"value": "canary"}}, "api_key"),
            ({"api_key": ["canary"]}, "api_key"),
            ({"api_key": 42}, "api_key"),
            (
                {"headers": {"Authorization": "Bearer canary"}},
                "headers.Authorization",
            ),
            (
                {"extra_headers": {"X-API-Key": "canary"}},
                "extra_headers.X-API-Key",
            ),
            (
                {"endpoint": "https://user:canary@example.test/v1"},
                "endpoint",
            ),
            (
                {"endpoint": "https://example.test/v1?auth_token=canary"},
                "endpoint",
            ),
            (
                {"endpoint": "https://example.test/v1?signature"},
                "endpoint",
            ),
        ],
    )
    def test_literal_router_credentials_never_enter_helm_values(
        self, tmp_path, config_document, expected_path
    ):
        canary = "literal-config-secret-canary"
        model = yaml.safe_load(
            yaml.safe_dump(config_document).replace("canary", canary)
        )
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "providers": {"models": [model]},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Kubernetes Router credential") as exc:
            translate_config_to_helm_values(str(config))

        assert f"providers.models[0].{expected_path}" in str(exc.value)
        assert canary not in str(exc.value)

    def test_environment_backed_and_empty_router_credentials_are_allowed(
        self, tmp_path
    ):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "listeners": [
                        {
                            "name": "public",
                            "port": 8000,
                            "api_keys": ["${LISTENER_API_KEY}"],
                        }
                    ],
                    "providers": {
                        "models": [
                            {"api_key": "${MODEL_API_KEY}"},
                            {"password": ""},
                            {"auth_token": None},
                            {
                                "api_key_env": "MODEL_API_KEY",
                                "auth_header": "Authorization",
                                "key_file": "/run/keys/public.pem",
                                "max_tokens": 128,
                                "headers": {
                                    "Authorization": "${AUTHORIZATION_HEADER}",
                                    "X-Tenant": "public-tenant",
                                },
                                "endpoint": "https://example.test/v1?version=1",
                            },
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        values = translate_config_to_helm_values(str(config))

        assert values["configOverride"]["providers"]["models"][0]["api_key"] == (
            "${MODEL_API_KEY}"
        )

    @pytest.mark.parametrize(
        ("credential_config", "expected_path"),
        [
            ({"api_key_env": "lowercase_name"}, "api_key_env"),
            ({"api_key_env": "HOME"}, "api_key_env"),
            ({"api_key_env": " MODEL_API_KEY "}, "api_key_env"),
            ({"api_key_env": {"value": "MODEL_API_KEY"}}, "api_key_env"),
            ({"api_key_env": ["MODEL_API_KEY"]}, "api_key_env"),
            ({"api_key_env": 42}, "api_key_env"),
            ({"api_key": "${PATH}"}, "api_key"),
            (
                {"headers": {"Authorization": "${HOME}"}},
                "headers.Authorization",
            ),
            ({"api_keys": ["${PATH}"]}, "api_keys[0]"),
        ],
    )
    def test_credential_environment_names_are_uppercase_and_non_reserved(
        self, tmp_path, credential_config, expected_path
    ):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "providers": {"models": [credential_config]},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="uppercase, non-reserved") as exc:
            translate_config_to_helm_values(str(config))

        assert f"providers.models[0].{expected_path}" in str(exc.value)
        assert "lowercase_name" not in str(exc.value)

    def test_listener_api_key_collection_requires_environment_references(
        self, tmp_path
    ):
        canary = "listener-secret-canary"
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "listeners": [
                        {
                            "name": "public",
                            "address": "0.0.0.0",
                            "port": 8000,
                            "api_keys": ["${LISTENER_API_KEY}", canary],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"listeners\[0\]\.api_keys\[1\]") as exc:
            translate_config_to_helm_values(str(config))

        assert canary not in str(exc.value)

    @pytest.mark.parametrize(
        "credential",
        [
            "${TOKEN:-literal-fallback}",
            "${TOKEN-literal-fallback}",
            "$BARE_TOKEN",
            " ${TOKEN}",
        ],
    )
    def test_only_pure_environment_credential_references_are_allowed(
        self, tmp_path, credential
    ):
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump({"version": "v0.3", "api_key": credential}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"config\.api_key") as exc:
            translate_config_to_helm_values(str(config))

        assert "fallback" not in str(exc.value)

    def test_write_helm_values_file(self, tmp_path):
        expected_replicas = 3
        values = {"replicaCount": expected_replicas, "image": {"tag": "v1"}}
        path = write_helm_values_file(values, str(tmp_path))

        written = yaml.safe_load(Path(path).read_text())
        assert written["replicaCount"] == expected_replicas
        assert stat.S_IMODE(Path(path).stat().st_mode) == 0o600

    def test_temporary_helm_values_file_is_removed_after_failure(self):
        canary = "inline-config-secret-canary"
        values_path = None
        values_dir = None

        with (
            pytest.raises(RuntimeError, match="synthetic Helm failure"),
            temporary_helm_values_file({"configOverride": {"api_key": canary}}) as path,
        ):
            values_path = Path(path)
            values_dir = values_path.parent
            assert canary in values_path.read_text(encoding="utf-8")
            assert stat.S_IMODE(values_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(values_dir.stat().st_mode) == 0o700
            raise RuntimeError("synthetic Helm failure")

        assert values_path is not None and not values_path.exists()
        assert values_dir is not None and not values_dir.exists()

    def test_sensitive_env_vars_excluded_from_plain_env(self, tmp_path):
        """Sensitive vars (masked=True in PASSTHROUGH_ENV_RULES) must not leak into plain env."""
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        env_vars = {
            "HF_TOKEN": "hf_test123",
            "OPENAI_API_KEY": "sk-test",
            "HF_ENDPOINT": "https://huggingface.co",
        }
        values = translate_config_to_helm_values(str(config), env_vars=env_vars)
        env_list = values.get("env", [])
        names = {e["name"] for e in env_list}
        assert "HF_TOKEN" not in names, "Sensitive var leaked into plain env"
        assert "OPENAI_API_KEY" not in names, "Sensitive var leaked into plain env"
        assert "HF_ENDPOINT" in names, "Non-sensitive var should be in plain env"

    def test_config_named_api_key_excluded_from_plain_env(self, tmp_path):
        """A key named only by api_key_env is still a credential and must not be inlined."""
        config = tmp_path / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "listeners": [],
                    "providers": {"models": [{"api_key_env": "GEMINI_API_KEY"}]},
                }
            )
        )

        values = translate_config_to_helm_values(
            str(config),
            env_vars={"GEMINI_API_KEY": "gk-test", "HF_ENDPOINT": "https://hf.co"},
        )
        names = {e["name"] for e in values.get("env", [])}
        leak_message = "Config-named credential leaked into plain env"
        assert "GEMINI_API_KEY" not in names, leak_message
        assert "HF_ENDPOINT" in names

    def test_router_log_level_env_vars_are_included_in_plain_env(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config),
            env_vars={"SR_LOG_LEVEL": "debug", "SR_LOG_ENCODING": "console"},
        )
        env_entries = {entry["name"]: entry["value"] for entry in values["env"]}

        assert env_entries["SR_LOG_LEVEL"] == "debug"
        assert env_entries["SR_LOG_ENCODING"] == "console"

    def test_env_secret_name_added_to_values(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(config), env_vars={"HF_ENDPOINT": "x"}, env_secret_name="my-secret"
        )
        assert "my-secret" in values["envFromSecrets"]

    def test_sensitivity_follows_source_config_when_effective_config_diverges(
        self, tmp_path
    ):
        """A name discovered from the source config must stay masked even if the
        effective config (algorithm/platform rewrite) no longer contains it."""
        source_config = tmp_path / "config.yaml"
        source_config.write_text(
            yaml.safe_dump(
                {"providers": {"models": [{"api_key_env": "GEMINI_API_KEY"}]}}
            )
        )
        # Simulates resolve_effective_config_path writing a rewritten copy that
        # dropped the api_key_env reference.
        effective_config = tmp_path / ".vllm-sr" / "runtime-config.yaml"
        effective_config.parent.mkdir(parents=True, exist_ok=True)
        effective_config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(
            str(effective_config),
            source_config_file=str(source_config),
            env_vars={"GEMINI_API_KEY": "gk-test"},
        )
        names = {e["name"] for e in values.get("env", [])}
        assert "GEMINI_API_KEY" not in names, (
            "credential leaked into plaintext Helm values because sensitivity was "
            "checked against the rewritten effective config instead of the source"
        )

    def test_env_vars_none_produces_no_env_key(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(yaml.safe_dump({"listeners": []}))

        values = translate_config_to_helm_values(str(config), env_vars=None)
        assert "env" not in values
        assert "envFromSecrets" not in values

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        overrides = {"a": {"b": 99}, "e": 4}
        merged = _deep_merge(base, overrides)
        assert merged == {"a": {"b": 99, "c": 2}, "d": 3, "e": 4}
