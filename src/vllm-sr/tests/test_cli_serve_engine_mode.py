"""Router and Engine use the same frontend and canonical active configuration."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import yaml
from cli import runtime_lifecycle
from cli.commands import runtime
from cli.commands.runtime_mode_config import (
    apply_instance_options,
    validate_model_options,
)
from cli.commands.runtime_serve_config import _prepare_docker_runtime_config
from cli.main import main
from cli.models import UserConfig
from cli.validator import validate_user_config
from click.testing import CliRunner

MODEL = "vllm-sr/Decision-2.0-Kai-0.6B"


def test_engine_and_router_use_the_same_serve_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime, "_execute_serve", lambda *args, **kw: calls.append((args, kw))
    )
    for mode in ("engine", "router"):
        result = CliRunner().invoke(
            main,
            ["serve", "--mode", mode, "--model", MODEL, "--device", "cpu", "--minimal"],
        )
        assert result.exit_code == 0, result.output
        assert calls[-1][1]["mode"] == mode
        assert calls[-1][1]["model_options"]["artifact"] == MODEL
    assert len(calls) == 2


def test_mode_switch_preserves_routing_and_existing_api_grants():
    document = {
        "version": "v0.3",
        "routing": {"decisions": [{"name": "saved"}]},
        "listeners": [{"name": "public", "models": ["router/auto"]}],
    }
    original = deepcopy(document)
    assert apply_instance_options(document, mode="engine")
    assert document["global"]["router"]["enabled"] is False
    assert document["routing"] == original["routing"]
    assert document["listeners"] == original["listeners"]
    assert not apply_instance_options(document)
    assert apply_instance_options(document, mode="router")
    assert document["global"]["router"]["enabled"] is True
    assert document["routing"] == original["routing"]


def test_first_engine_start_creates_a_minimal_canonical_instance():
    document = {"version": "v0.3", "setup": {"mode": True}}
    resource = validate_model_options(
        model=MODEL, revision=None, device="cpu", runtime_profile=None, platform="cpu"
    )
    assert apply_instance_options(document, mode="engine", model_options=resource)
    assert "setup" not in document
    assert "providers" not in document and "routing" not in document
    assert document["listeners"][0]["systemone"]["models"] == [MODEL]
    assert document["global"]["model_catalog"]["system"]["decision_model"] == {
        "deployment": "primary"
    }
    assert (
        validate_user_config(UserConfig.model_validate(document), log_summary=False)
        == []
    )


def test_model_replacement_keeps_explicit_public_identity_and_grants():
    document = {
        "global": {
            "model_catalog": {
                "deployments": {
                    "chosen": {
                        "provider": "model_runtime",
                        "artifact": "old/model",
                        "public_name": "judge",
                        "replicas": [{"device": "rocm:0"}, {"device": "rocm:1"}],
                    }
                }
            }
        },
        "listeners": [{"name": "main", "systemone": {"models": ["judge"]}}],
    }
    resource = validate_model_options(
        model=MODEL, revision=None, device="cpu", runtime_profile=None, platform="cpu"
    )
    apply_instance_options(document, model_options=resource, decision_model="chosen")
    changed = document["global"]["model_catalog"]["deployments"]["chosen"]
    assert changed["public_name"] == "judge"
    assert changed["artifact"] == MODEL
    assert "replicas" not in changed
    assert document["listeners"][0]["systemone"]["models"] == ["judge"]


@pytest.mark.parametrize(
    "args,message",
    [
        ([MODEL], "unexpected extra argument"),
        (["--models", "models.yaml"], "No such option"),
        (["--device", "cpu"], "requires --model"),
        (["--model", MODEL, "--revision", "main"], "40-hex"),
        (["--model", MODEL, "--device", "cuda"], "needs --platform nvidia"),
        (["--model", MODEL, "--runtime-profile", "Fast!"], "profile name"),
    ],
)
def test_invalid_shortcuts_fail_before_startup(monkeypatch, args, message):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid command reached startup")

    monkeypatch.setattr(runtime, "_execute_serve", unexpected)
    result = CliRunner().invoke(main, ["serve", *args])
    assert result.exit_code == 2, result.output
    assert message in result.output


def test_help_describes_capability_mode_and_explicit_model_flag():
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "INSTANCE MODES" in result.output
    assert "--mode" in result.output and "--model" in result.output
    assert "vllm-sr serve MODEL" not in result.output


@pytest.mark.parametrize("enabled", ["false", 0, None])
def test_routing_enabled_is_strictly_boolean(enabled):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        UserConfig.model_validate(
            {"version": "v0.3", "global": {"router": {"enabled": enabled}}}
        )


def test_active_mode_changes_preserve_source_and_survive_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime_lifecycle, "container_status_strict", lambda _: "exited"
    )
    monkeypatch.setattr(runtime_lifecycle, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        runtime_lifecycle, "acquire_runtime_lifecycle_lock", lambda **_: nullcontext()
    )
    source = tmp_path / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [{"name": "private", "address": "0.0.0.0", "port": 8899}],
                "routing": {"strategy": "confidence"},
                "global": {
                    "services": {"observability": {"tracing": {"enabled": False}}}
                },
            }
        )
    )
    before = source.read_bytes()

    def prepare(mode=None):
        path, setup, lock = _prepare_docker_runtime_config(
            source,
            None,
            False,
            None,
            (),
            False,
            mode=mode,
        )
        lock.close()
        assert not setup
        return yaml.safe_load(path.read_text())

    engine = prepare("engine")
    assert engine["global"]["router"]["enabled"] is False
    assert engine["routing"]["strategy"] == "confidence"
    assert "systemone" not in engine["listeners"][0]
    assert prepare()["global"]["router"]["enabled"] is False
    assert prepare("router")["global"]["router"]["enabled"] is True
    assert source.read_bytes() == before
