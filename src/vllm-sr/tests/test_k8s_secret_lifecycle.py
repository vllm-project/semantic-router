"""Tests for K8s backend discovery and runtime Secret planning/creation."""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _k8s_test_helpers import (  # noqa: E402
    k8s_backend,
    managed_secret_payload,
)
from cli.k8s_backend import K8sBackend  # noqa: E402


class TestK8sBackend:
    def test_require_tool_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(SystemExit):
            K8sBackend._require_tool("helm")

    def test_helm_base_cmd_includes_context(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = "my-ctx"
        assert backend._helm_base_cmd() == ["helm", "--kube-context", "my-ctx"]

    def test_helm_base_cmd_without_context(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.context = None
        assert backend._helm_base_cmd() == ["helm"]

    def test_existing_namespace_is_never_applied_or_updated(self, monkeypatch):
        backend = k8s_backend()
        monkeypatch.setattr(backend, "_namespace_exists", lambda: True)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: pytest.fail(
                "an existing Namespace must not require cluster-scoped update access"
            ),
        )

        K8sBackend._ensure_namespace(backend)

    def test_absent_namespace_uses_create_only_and_accepts_creation_race(
        self, monkeypatch
    ):
        backend = k8s_backend()
        states = iter([False, True])
        calls = []
        monkeypatch.setattr(backend, "_namespace_exists", lambda: next(states))
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: calls.append((cmd, kwargs))
            or MagicMock(returncode=1),
        )

        K8sBackend._ensure_namespace(backend)

        assert calls[0][0][1:] == ["create", "namespace", "shared-ns"]
        assert calls[0][1]["check"] is False
        assert "apply" not in calls[0][0]

    def test_namespace_create_failure_is_verified_and_fails_closed(self, monkeypatch):
        backend = k8s_backend()
        monkeypatch.setattr(backend, "_namespace_exists", lambda: False)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: MagicMock(returncode=1),
        )

        with pytest.raises(RuntimeError, match="namespace creation failed"):
            K8sBackend._ensure_namespace(backend)

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ("[]", False),
            ('[{"name":"alpha","status":"deployed"}]', True),
        ],
    )
    def test_helm_release_state_requires_exact_structured_match(
        self, monkeypatch, payload, expected
    ):
        backend = k8s_backend()
        calls = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **kwargs: (
                calls.append((cmd, kwargs))
                or MagicMock(returncode=0, stdout=payload, stderr="")
            ),
        )

        assert backend._helm_release_exists() is expected
        assert calls[0][0][1:3] == ["list", "--all"]
        assert calls[0][1]["capture_output"] is True

    @pytest.mark.parametrize(
        "result",
        [
            MagicMock(returncode=1, stdout="", stderr="sensitive-server-output"),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="not-json", stderr=""),
            MagicMock(returncode=0, stdout="{}", stderr=""),
            MagicMock(
                returncode=0,
                stdout='[{"name":"beta","status":"deployed"}]',
                stderr="",
            ),
            MagicMock(returncode=0, stdout='[{"name":"alpha"}]', stderr=""),
        ],
    )
    def test_helm_release_state_fails_closed_without_output_leak(
        self, monkeypatch, caplog, result
    ):
        backend = k8s_backend()
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: result)

        with pytest.raises(RuntimeError, match="Helm"):
            backend._helm_release_exists()

        assert "sensitive-server-output" not in caplog.text

    def test_label_for_service_router(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "semantic-router" in backend._label_for_service("router")

    def test_label_for_service_all(self):
        backend = K8sBackend.__new__(K8sBackend)
        backend.release_name = "sr"
        assert "sr" in backend._label_for_service("all")

    def test_secret_plan_generates_looper_key_for_empty_inputs(self, monkeypatch):
        backend = k8s_backend()
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])

        for env_vars in (None, {}, {"HF_ENDPOINT": "https://hf.co"}):
            plan = backend._plan_env_secret(env_vars)
            data = json.loads(plan.new_manifest)["data"]
            generated = base64.b64decode(data["VLLM_SR_LOOPER_SHARED_SECRET"]).decode()
            assert len(generated) == 64 and int(generated, 16) >= 0
            assert plan.creates_secret is True
            assert plan.recreate_for_looper_rotation is True

    def test_create_uses_immutable_stdin_manifest_without_argv_leak(
        self, monkeypatch, caplog
    ):
        backend = k8s_backend()
        calls = []
        secret_value = "hf_secret123"
        generated_looper = "a" * 64
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            "cli.k8s_backend.secrets.token_hex",
            lambda size: generated_looper if size == 32 else "b" * 12,
        )
        monkeypatch.setattr(backend, "_ensure_namespace", lambda: None)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: (
                calls.append((cmd, kwargs)) or MagicMock(returncode=0)
            ),
        )

        plan = backend._plan_env_secret({"HF_TOKEN": secret_value})
        created_name = backend._create_planned_secret(plan)

        assert created_name == plan.active_name
        assert plan.active_name.startswith("alpha-vsr-env-")
        assert secret_value not in repr(plan)
        cmd, kwargs = calls[0]
        assert cmd[-3:] == ["create", "-f", "-"]
        manifest = json.loads(kwargs["input_text"])
        assert manifest["immutable"] is True
        assert manifest["data"] == backend._encode_secret_data(
            {
                "HF_TOKEN": secret_value,
                "VLLM_SR_LOOPER_SHARED_SECRET": generated_looper,
            }
        )
        assert manifest["metadata"]["labels"]["app.kubernetes.io/instance"] == "alpha"
        leaked = [secret_value, generated_looper, *manifest["data"].values()]
        assert all(
            value not in " ".join(cmd) and value not in caplog.text for value in leaked
        )

    def test_empty_explicit_looper_secret_is_rejected(self, monkeypatch):
        backend = k8s_backend()
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])

        with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
            backend._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": ""})

    def test_unchanged_exact_data_reuses_immutable_generation(self, monkeypatch):
        backend = k8s_backend()
        current_name = "alpha-vsr-env-current"
        payload = managed_secret_payload(
            current_name,
            "alpha",
            {
                "HF_TOKEN": "same",
                "OPENAI_API_KEY": "same-key",
                "VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64,
            },
        )
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)

        plan = backend._plan_env_secret(
            {"HF_TOKEN": "same", "OPENAI_API_KEY": "same-key"}
        )

        assert plan.active_name == current_name
        assert plan.creates_secret is False
        assert plan.new_manifest is None
        assert plan.recreate_for_looper_rotation is True

    def test_exact_key_set_change_creates_new_generation(self, monkeypatch):
        backend = k8s_backend()
        current_name = "alpha-vsr-env-current"
        payload = managed_secret_payload(
            current_name,
            "alpha",
            {
                "HF_TOKEN": "same",
                "OPENAI_API_KEY": "removed",
                "VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64,
            },
        )
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: [current_name]
        )
        monkeypatch.setattr(backend, "_get_secret_json", lambda _: payload)

        plan = backend._plan_env_secret({"HF_TOKEN": "same"})
        manifest = json.loads(plan.new_manifest)

        assert plan.active_name != current_name
        assert manifest["data"] == backend._encode_secret_data(
            {"HF_TOKEN": "same", "VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64}
        )
        assert plan.recreate_for_looper_rotation is True

    def test_release_scoped_secrets_do_not_cross_releases(self, monkeypatch):
        alpha = k8s_backend("alpha")
        beta = k8s_backend("beta")
        monkeypatch.setattr(alpha, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(beta, "_current_router_secret_refs", lambda: [])

        alpha_plan = alpha._plan_env_secret({"HF_TOKEN": "alpha-token"})
        beta_plan = beta._plan_env_secret({"HF_TOKEN": "beta-token"})
        alpha_manifest = json.loads(alpha_plan.new_manifest)
        beta_manifest = json.loads(beta_plan.new_manifest)

        assert alpha_plan.active_name != beta_plan.active_name
        assert (
            alpha_manifest["metadata"]["labels"]["app.kubernetes.io/instance"]
            == "alpha"
        )
        assert (
            beta_manifest["metadata"]["labels"]["app.kubernetes.io/instance"] == "beta"
        )
