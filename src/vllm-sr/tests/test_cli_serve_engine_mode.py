"""One frontend with startup-only routing and explicit model placement overrides."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import yaml
from cli import decision_model, execution_platform, runtime_lifecycle
from cli.bootstrap import build_bootstrap_config
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


def options(model=None, **kwargs):
    return validate_model_options(
        model=model,
        revision=kwargs.pop("revision", None),
        runtime_profile=kwargs.pop("runtime_profile", None),
        **kwargs
    )


@pytest.mark.parametrize(
    "args,engine",
    [([], False), ([MODEL], False), ([MODEL, "-e"], True), (["--engine"], True)],
)
def test_router_default_and_explicit_engine_share_serve(monkeypatch, args, engine):
    calls = []
    monkeypatch.setattr(runtime, "_execute_serve", lambda *a, **kw: calls.append(kw))
    result = CliRunner().invoke(main, ["serve", *args])
    assert result.exit_code == 0, result.output
    assert calls[0]["engine"] is engine
    if MODEL in args:
        assert calls[0]["model_options"] == {"artifact": MODEL}


def test_no_engine_flag_enables_routing_without_discarding_saved_policy():
    document = {
        "version": "v0.3",
        "routing": {"decisions": [{"name": "saved"}]},
        "listeners": [{"name": "public", "models": ["router/auto"]}],
    }
    original = deepcopy(document)
    assert apply_instance_options(document, engine=True)
    assert document["global"]["router"]["enabled"] is False
    assert apply_instance_options(document)
    assert document["global"]["router"]["enabled"] is True
    assert not apply_instance_options(document)
    assert document["routing"] == original["routing"]
    assert document["listeners"] == original["listeners"]


def test_engine_bootstrap_has_native_grant_and_no_invented_provider():
    document = {"version": "v0.3", "setup": {"mode": True}}
    apply_instance_options(document, engine=True, model_options=options(MODEL))
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


def test_generated_engine_bootstrap_preserves_listener_settings():
    document = build_bootstrap_config()
    original = deepcopy(document["listeners"][0])
    apply_instance_options(document, engine=True, model_options=options(MODEL))
    assert document["listeners"] == [{**original, "systemone": {"models": [MODEL]}}]


@pytest.mark.parametrize(
    "overrides",
    [
        {"address": "127.0.0.1", "port": 9999},
        {"api_keys": ["private-test-key"]},
        {"models": ["vllm-sr/auto"]},
        {"systemone": {"models": [MODEL]}},
        {
            "address": "127.0.0.1",
            "port": 9999,
            "api_keys": ["private-test-key"],
            "systemone": {"models": [MODEL]},
        },
    ],
)
def test_engine_setup_preserves_authored_listener_policy(overrides):
    document = build_bootstrap_config()
    document["listeners"][0].update(overrides)
    listeners = deepcopy(document["listeners"])
    apply_instance_options(document, engine=True, model_options=options(MODEL))
    assert "setup" not in document
    assert document["global"]["router"]["enabled"] is False
    assert document["listeners"] == listeners
    assert (
        validate_user_config(UserConfig.model_validate(document), log_summary=False)
        == []
    )


def declaration(resource):
    return {
        "global": {
            "model_catalog": {
                "system": {"decision_model": {"deployment": "chosen"}},
                "deployments": {"chosen": resource},
            }
        },
        "listeners": [{"name": "main", "systemone": {"models": ["judge"]}}],
    }


def selected(document):
    return document["global"]["model_catalog"]["deployments"]["chosen"]


def test_model_only_keeps_placement_profile_identity_and_grants():
    replicas = [{"device": "rocm:0"}, {"device": "rocm:1"}]
    document = declaration(
        {
            "provider": "model_runtime",
            "artifact": "old/model",
            "revision": "a" * 40,
            "public_name": "judge",
            "profile": "batching",
            "replicas": replicas,
            "input": {"max_tokens": 4096, "overflow": "window"},
        }
    )
    apply_instance_options(document, model_options=options(MODEL), platform="rocm")
    result = selected(document)
    assert result["artifact"] == MODEL and result["revision"] != "a" * 40
    assert result["replicas"] == replicas and result["profile"] == "batching"
    assert result["input"] == {"max_tokens": 4096, "overflow": "window"}
    assert result["public_name"] == "judge"
    assert document["listeners"][0]["systemone"]["models"] == ["judge"]


def test_same_artifact_keeps_revision_and_profile_when_unspecified():
    document = declaration(
        {
            "provider": "model_runtime",
            "artifact": MODEL,
            "revision": "b" * 40,
            "profile": "batching",
            "device": "cpu",
        }
    )
    apply_instance_options(document, model_options=options(MODEL))
    assert selected(document)["revision"] == "b" * 40
    assert selected(document)["profile"] == "batching"


def test_new_multi_gpu_workers_use_available_devices():
    document = {"version": "v0.3"}
    apply_instance_options(
        document,
        model_options=options(MODEL, data_parallel_size=2),
        platform="rocm",
        available_devices=(2, 5),
    )
    result = document["global"]["model_catalog"]["deployments"]["primary"]
    assert result["replicas"] == [{"device": "rocm:2"}, {"device": "rocm:5"}]
    assert "device" not in result
    assert result["profile"] == "exact"


def test_new_dp_requires_explicit_colocation_when_capacity_is_short():
    with pytest.raises(ValueError, match="available GPUs"):
        apply_instance_options(
            {},
            model_options=options(MODEL, data_parallel_size=4),
            platform="rocm",
            available_devices=(0,),
        )


def test_explicit_host_device_can_repeat_workers_after_mask_mapping():
    requested = options(MODEL, data_parallel_size=4, device_ids="7")
    requested["device_ordinals"] = (0,)
    document = {}
    apply_instance_options(
        document, model_options=requested, platform="rocm", available_devices=(0,)
    )
    result = document["global"]["model_catalog"]["deployments"]["primary"]
    assert result["replicas"] == [{"device": "rocm:0"}] * 4
    assert "data_parallel_size" not in result and "device_ids" not in result


def test_dp_scaling_preserves_existing_placement_order():
    document = declaration(
        {
            "provider": "model_runtime",
            "artifact": MODEL,
            "replicas": [{"device": "cuda:1"}, {"device": "cuda:3"}],
        }
    )
    apply_instance_options(
        document, model_options=options(data_parallel_size=3), platform="cuda"
    )
    assert selected(document)["replicas"] == [
        {"device": "cuda:1"},
        {"device": "cuda:3"},
        {"device": "cuda:1"},
    ]


def test_attached_deployment_is_not_silently_replaced():
    document = declaration(
        {
            "provider": "model_runtime",
            "artifact": MODEL,
            "endpoint": "http://runtime:8100",
        }
    )
    with pytest.raises(ValueError, match="managed default deployment"):
        apply_instance_options(document, model_options=options(MODEL))


@pytest.mark.parametrize(
    "args",
    [
        ["--mode", "engine"],
        ["--model", MODEL],
        ["--decision-model", "primary"],
        ["--device", "cpu"],
        ["--models", "models.yaml"],
        ["--platform", "amd"],
        ["--platform", "nvidia"],
        ["-dp", "0"],
        ["-dp", "65"],
        ["--device-ids", "0,0"],
        ["--device-ids", "-1"],
        ["--runtime-profile", "Fast!"],
    ],
)
def test_removed_or_invalid_options_never_reach_startup(monkeypatch, args):
    monkeypatch.setattr(
        runtime,
        "_execute_serve",
        lambda *a, **kw: pytest.fail("invalid command reached startup"),
    )
    result = CliRunner().invoke(main, ["serve", *args])
    assert result.exit_code == 2, result.output


def test_help_and_instance_commands_have_no_mode_bypass():
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "[MODEL]" in result.output and "--engine" in result.output
    assert (
        "--model " not in result.output
        and "--decision-model" not in result.output
        and "--mode " not in result.output
    )
    assert "--data-parallel-size" in result.output and "--device-ids" in result.output
    help_result = CliRunner().invoke(main, ["instance", "--help"])
    assert (
        "deploy" not in help_result.output
        and "controller" not in help_result.output
        and "attach" not in help_result.output
    )


@pytest.mark.parametrize("enabled", ["false", 0, None])
def test_routing_enabled_is_strictly_boolean(enabled):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        UserConfig.model_validate(
            {"version": "v0.3", "global": {"router": {"enabled": enabled}}}
        )


def test_startup_mode_preserves_source_and_resets_engine_without_flag(
    tmp_path, monkeypatch
):
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

    def prepare(engine=False):
        path, setup, lock = _prepare_docker_runtime_config(
            source, None, False, "cpu", (), False, engine=engine
        )
        lock.close()
        assert not setup
        return yaml.safe_load(path.read_text())

    assert prepare(True)["global"]["router"]["enabled"] is False
    router = prepare()
    assert router["global"]["router"]["enabled"] is True
    assert router["routing"]["strategy"] == "confidence"
    assert "systemone" not in router["listeners"][0]
    assert source.read_bytes() == before


def test_serve_projects_host_ids_once_and_forwards_the_existing_mask(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        execution_platform,
        "_host_devices",
        lambda: {"cuda": (), "rocm": tuple(range(8))},
    )
    monkeypatch.setenv("VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES", "7,3")
    monkeypatch.setattr(decision_model, "host_has_gpu", lambda _: True)
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
                "listeners": [
                    {"name": "private", "address": "127.0.0.1", "port": 8899}
                ],
                "global": {
                    "services": {"observability": {"tracing": {"enabled": False}}}
                },
            }
        )
    )
    before = source.read_bytes()
    monkeypatch.setattr(runtime, "_resolve_serve_config", lambda *args: (source, False))
    deployed = {}
    monkeypatch.setattr(
        runtime, "_deploy_serve_backend", lambda **kwargs: deployed.update(kwargs)
    )
    result = CliRunner().invoke(
        main,
        [
            "serve",
            MODEL,
            "-e",
            "--platform",
            "rocm",
            "-dp",
            "2",
            "--device-ids",
            "3,7",
            "--config",
            str(source),
        ],
    )
    assert result.exit_code == 0, result.output
    document = yaml.safe_load(deployed["effective_config_path"].read_text())
    primary = document["global"]["model_catalog"]["deployments"]["primary"]
    assert primary["replicas"] == [{"device": "rocm:1"}, {"device": "rocm:0"}]
    assert len(primary["revision"]) == 40
    assert deployed["env_vars"]["ROCR_VISIBLE_DEVICES"] == "7,3"
    assert document["global"]["router"]["enabled"] is False
    assert "systemone" not in document["listeners"][0]
    assert source.read_bytes() == before


def test_resource_without_authored_placement_uses_available_gpus():
    document = declaration({"provider": "model_runtime", "artifact": MODEL})
    apply_instance_options(
        document,
        model_options=options(data_parallel_size=2),
        platform="cuda",
        available_devices=(1, 3),
    )
    assert selected(document)["replicas"] == [
        {"device": "cuda:1"},
        {"device": "cuda:3"},
    ]
