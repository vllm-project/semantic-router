"""`vllm-sr serve --decision-model`, its config version, status and validation."""

import json
import subprocess
from contextlib import nullcontext

import pytest
import yaml
from cli import decision_model, runtime_lifecycle, runtime_service_status
from cli.commands import runtime_config_mutation, runtime_paths, runtime_serve_config
from cli.k8s_backend import K8sBackend
from cli.main import main
from cli.models import UserConfig
from cli.runtime_stack import resolve_runtime_stack
from cli.validator import validate_user_config
from click.testing import CliRunner

SOURCE = (
    "version: v0.3\nlisteners:\n- name: main\n  address: 0.0.0.0\n  port: 8888\n"
    "global:\n  services:\n    observability:\n      tracing:\n        enabled: false\n"
)


def _system(document):
    return document["global"]["model_catalog"]["system"]


@pytest.mark.parametrize(
    "name, canonical",
    [
        (None, "Vela-2.0-0.3B"),
        ("", "Vela-2.0-0.3B"),
        ("vela-2.0-0.8b", "Vela-2.0-0.8B"),
        (" VELA-2.0-9B ", "Vela-2.0-9B"),
        ("vela-1.0", "Vela-1.0"),
    ],
)
def test_names_are_canonical_and_case_insensitive(name, canonical):
    assert decision_model.canonical_decision_model(name) == canonical


@pytest.mark.parametrize(
    "name, message",
    [
        ("Vela-2.0-27B", "is not a decision model; choose Vela-2.0-0.3B (the default)"),
        ("vllm-sr/Vela-2.0-4B", "name the model Vela-2.0-4B, without a repository"),
        ("Decision-2.0-Eos-0.8B", "is a Decision 2.0 model"),
        ("Lux", "name that deployment in a routing.signals.decision question"),
    ],
)
def test_other_names_are_rejected_with_the_routers_wording(name, message):
    with pytest.raises(ValueError, match="decision_model") as error:
        decision_model.canonical_decision_model(name)
    assert message in str(error.value)


def test_the_4b_and_9b_need_a_gpu(monkeypatch):
    monkeypatch.setattr(decision_model, "host_has_gpu", lambda platform: False)
    for name in ("Vela-2.0-4B", "vela-2.0-9b"):
        error = decision_model.gpu_requirement_error(name, "", local_host=True)
        assert "needs a GPU; serve it with --platform amd or --platform nvidia" in error
        error = decision_model.gpu_requirement_error(name, "amd", local_host=True)
        assert "this host shows no AMD GPU devices" in error
        # A cluster's GPUs are not this host's: only the platform is checked.
        assert (
            decision_model.gpu_requirement_error(name, "amd", local_host=False) is None
        )
    monkeypatch.setattr(decision_model, "host_has_gpu", lambda platform: True)
    assert (
        decision_model.gpu_requirement_error("Vela-2.0-9B", "nvidia", local_host=True)
        is None
    )
    for name in ("Vela-2.0-0.3B", "Vela-2.0-0.8B", "Vela-1.0"):
        assert decision_model.gpu_requirement_error(name, "", local_host=True) is None


def test_engine_mode_rejects_the_flag():
    result = CliRunner().invoke(
        main, ["serve", "vllm-sr/Vela-2.0-0.3B", "--decision-model", "Vela-2.0-0.8B"]
    )
    assert result.exit_code == 2
    assert "--decision-model applies to the Router stack" in result.output


def test_serve_refuses_a_gpu_model_on_cpu_before_anything_changes(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(SOURCE)
    result = CliRunner().invoke(
        main, ["serve", "--config", str(config), "--decision-model", "Vela-2.0-9B"]
    )
    assert result.exit_code != 0
    assert "the decision model Vela-2.0-9B needs a GPU" in result.output
    assert config.read_text() == SOURCE
    assert not (tmp_path / ".vllm-sr").exists()


def _quiet_lifecycle(monkeypatch):
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda name: "exited"
    )
    monkeypatch.setattr(runtime_lifecycle, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_lifecycle,
        "acquire_runtime_lifecycle_lock",
        lambda **kwargs: nullcontext(),
    )


def _prepare(source, decision=None, replace=False):
    path, _setup, lock = runtime_serve_config._prepare_docker_runtime_config(
        source, None, False, None, (), replace, decision_model=decision
    )
    lock.close()
    return path


def test_the_flag_writes_a_config_version_that_later_starts_keep(tmp_path, monkeypatch):
    _quiet_lifecycle(monkeypatch)
    source = tmp_path / "config.yaml"
    source.write_text(SOURCE)
    active = _prepare(source)
    assert "decision_model" not in active.read_text()

    assert _prepare(source, "Vela-2.0-0.8B") == active
    assert _system(yaml.safe_load(active.read_text())) == {
        "decision_model": "Vela-2.0-0.8B"
    }
    assert source.read_text() == SOURCE, "the user's config is never rewritten"

    # A later start without the flag keeps the active config's decision model.
    _prepare(source)
    assert (
        _system(yaml.safe_load(active.read_text()))["decision_model"] == "Vela-2.0-0.8B"
    )
    # Asking for the active model writes nothing.
    before = active.read_bytes()
    _prepare(source, "vela-2.0-0.8b")
    assert active.read_bytes() == before
    # --replace-active-config returns to the source config.
    _prepare(source, replace=True)
    assert "decision_model" not in active.read_text()


def test_a_later_start_on_cpu_refuses_an_active_gpu_model(tmp_path, monkeypatch):
    _quiet_lifecycle(monkeypatch)
    source = tmp_path / "config.yaml"
    source.write_text(SOURCE)
    active = _prepare(source)
    document = yaml.safe_load(active.read_text())
    decision_model.set_decision_model(document, "Vela-2.0-4B")
    runtime_paths._atomic_write_private_bytes(active, yaml.safe_dump(document).encode())
    with pytest.raises(
        ValueError, match=r"the decision model Vela-2\.0-4B needs a GPU"
    ):
        _prepare(source)


def _validate(document):
    return [
        str(error)
        for error in validate_user_config(UserConfig(**document), log_summary=False)
    ]


QUESTION = {
    "name": "needs_tools",
    "question": {"type": "noul", "instructions": "Does the request need a tool?"},
}


def _document(system=None):
    document = yaml.safe_load(SOURCE)
    document["providers"] = {
        "defaults": {"model": "small"},
        "models": [{"name": "small", "backend_refs": [{"endpoint": "small:8000"}]}],
    }
    document["routing"] = {
        "modelCards": [{"name": "small"}],
        "signals": {"decision": [dict(QUESTION)]},
        "decisions": [
            {
                "name": "tools",
                "priority": 10,
                "rules": {
                    "operator": "AND",
                    "conditions": [{"type": "decision", "name": "needs_tools"}],
                },
                "modelRefs": [{"model": "small"}],
            }
        ],
    }
    if system is not None:
        document["global"]["model_catalog"] = {"system": system}
    return document


def test_config_validate_checks_the_decision_model():
    assert _validate(_document()) == []
    assert _validate(_document({"decision_model": "vela-2.0-4b"})) == []
    errors = _validate(_document({"decision_model": "Decision-2.0-Nox-4B"}))
    assert any(
        error.startswith("[global.model_catalog.system.decision_model]")
        and "is a Decision 2.0 model" in error
        for error in errors
    ), errors
    errors = _validate(_document({"decision_model": "Vela-1.0"}))
    assert any(
        "routing.signals.decision.needs_tools" in error
        and "deployment is required: the decision model is Vela-1.0" in error
        for error in errors
    ), errors


def test_status_reads_the_routers_active_decision_model(monkeypatch):
    document = {
        "global": {"model_catalog": {"system": {"decision_model": "vela-2.0-9b"}}}
    }
    monkeypatch.setattr(
        runtime_service_status,
        "container_exec",
        lambda name, command: (0, yaml.safe_dump(document), ""),
    )
    assert (
        runtime_service_status.active_decision_model(resolve_runtime_stack())
        == "Vela-2.0-9B"
    )
    shown = []
    monkeypatch.setattr(runtime_service_status, "_check_router_status", lambda _: True)
    monkeypatch.setattr(runtime_service_status, "fields", shown.extend)
    runtime_service_status.report_service_status("router", resolve_runtime_stack())
    assert ("Decision model", "Vela-2.0-9B") in shown
    monkeypatch.setattr(
        runtime_service_status, "container_exec", lambda name, command: (1, "", "")
    )
    assert runtime_service_status.active_decision_model(resolve_runtime_stack()) is None


def test_kubernetes_status_reads_the_live_config(monkeypatch, tmp_path):
    live = {
        "global": {"model_catalog": {"system": {"decision_model": "Vela-2.0-0.8B"}}}
    }
    items = [
        {"data": {"policy.yaml": "x"}},
        {"data": {"config.yaml": yaml.safe_dump(live), "tools_db.json": "[]"}},
    ]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, json.dumps({"items": items}), ""
        ),
    )
    backend = K8sBackend(namespace="vllm-sr", chart_dir=str(tmp_path))
    assert backend._live_decision_model() == "Vela-2.0-0.8B"


def test_gpu_platforms_put_the_safety_module_on_the_gpu(monkeypatch):
    monkeypatch.delenv("VLLM_SR_AMD_PRESERVE_CPU", raising=False)
    monkeypatch.delenv("VLLM_SR_AMD_FORCE_GPU", raising=False)
    config = {
        "global": {
            "model_catalog": {"modules": {"safety": {"safety": {"use_cpu": True}}}}
        }
    }
    assert runtime_config_mutation.apply_platform_gpu_defaults(config, "amd")
    assert (
        config["global"]["model_catalog"]["modules"]["safety"]["safety"]["use_cpu"]
        is False
    )
