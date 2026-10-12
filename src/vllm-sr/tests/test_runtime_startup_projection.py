"""Startup overrides retain resources without claiming operator edits as source."""

import hashlib
import json

import pytest
import yaml
from cli.commands import runtime_paths, runtime_serve_config
from cli.commands.runtime_mode_config import apply_instance_options

MODEL = "vllm-sr/Decision-2.0-Kai-0.6B"


@pytest.fixture
def startup(tmp_path, monkeypatch):
    source = tmp_path / "config.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "version": "v0.3",
                "listeners": [
                    {"name": "private", "address": "127.0.0.1", "port": 8899}
                ],
                "routing": {"strategy": "confidence"},
                "global": {
                    "services": {"observability": {"tracing": {"enabled": False}}}
                },
            }
        )
    )
    monkeypatch.setattr(
        runtime_serve_config, "stop_runtime_before_config_replacement", lambda _: None
    )

    def prepare(**kwargs):
        path, _, lock = runtime_serve_config._prepare_docker_runtime_config(
            source, None, False, "cpu", (), False, **kwargs
        )
        lock.close()
        return path

    return source, prepare


def assert_owned(path, expected=True):
    receipt = json.loads(
        runtime_paths._runtime_config_provenance_path(path).read_text()
    )
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert (receipt["last_materialized_active_digest"] == digest) is expected


def test_router_defaults_do_not_modify_authored_document():
    document = {"version": "v0.3"}
    assert not apply_instance_options(document)
    assert document == {"version": "v0.3"}


def test_gpu_platform_materializes_meaningful_default_placement():
    document = {"version": "v0.3"}
    assert apply_instance_options(document, platform="rocm")
    primary = document["global"]["model_catalog"]["deployments"]["primary"]
    assert primary["device"] == "auto"
    assert len(primary["revision"]) == 40


def test_same_source_restart_retains_resolved_model_profile_and_replicas(startup):
    source, prepare = startup
    authored = source.read_bytes()
    path = prepare(
        engine=True,
        model_options={"artifact": MODEL, "profile": "exact", "data_parallel_size": 2},
    )
    chosen = yaml.safe_load(path.read_text())["global"]["model_catalog"]
    assert_owned(path)
    prepare()
    restarted = yaml.safe_load(path.read_text())
    assert restarted["global"]["router"]["enabled"] is True
    assert restarted["global"]["model_catalog"] == chosen
    assert source.read_bytes() == authored
    assert_owned(path)


@pytest.mark.parametrize("replacement_model", [None, "vllm-sr/Vela-2.0-0.3B"])
def test_source_change_replaces_old_cli_selection_without_three_way_merge(
    startup, replacement_model
):
    source, prepare = startup
    path = prepare(model_options={"artifact": MODEL, "data_parallel_size": 2})
    changed = yaml.safe_load(source.read_text())
    changed["routing"]["strategy"] = "priority"
    if replacement_model:
        changed["global"]["model_catalog"] = {
            "system": {"decision_model": {"deployment": "chosen"}},
            "deployments": {
                "chosen": {
                    "provider": "model_runtime",
                    "artifact": replacement_model,
                    "device": "cpu",
                    "profile": "exact",
                }
            },
        }
    source.write_text(yaml.safe_dump(changed))
    prepare()
    current = yaml.safe_load(path.read_text())
    assert current["routing"] == changed["routing"]
    assert current["listeners"] == changed["listeners"]
    assert current["global"].get("model_catalog") == changed["global"].get(
        "model_catalog"
    )
    assert_owned(path)


def test_mode_projection_keeps_dashboard_edit_ownership(startup):
    source, prepare = startup
    path = prepare()
    edited = yaml.safe_load(path.read_text())
    edited["routing"]["strategy"] = "priority"
    path.write_text(yaml.safe_dump(edited))
    prepare(engine=True)
    assert yaml.safe_load(path.read_text())["routing"]["strategy"] == "priority"
    assert_owned(path, False)
    changed = yaml.safe_load(source.read_text())
    changed["listeners"][0]["port"] = 8898
    source.write_text(yaml.safe_dump(changed))
    prepare()
    assert yaml.safe_load(path.read_text())["routing"]["strategy"] == "priority"
    assert_owned(path, False)


@pytest.mark.parametrize("user_edited", [False, True])
def test_interrupted_startup_write_recovers_only_exact_cli_projection(
    startup, monkeypatch, user_edited
):
    source, prepare = startup
    path = prepare()
    provenance_path = runtime_paths._runtime_config_provenance_path(path)
    write = runtime_paths.write_private_state_bytes

    def fail_provenance(target, data, **kwargs):
        if target == provenance_path:
            raise OSError("interrupted startup provenance")
        return write(target, data, **kwargs)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(runtime_paths, "write_private_state_bytes", fail_provenance)
        with pytest.raises(OSError, match="interrupted startup provenance"):
            prepare(engine=True)
    assert yaml.safe_load(path.read_text())["global"]["router"]["enabled"] is False
    if user_edited:
        document = yaml.safe_load(path.read_text())
        document["routing"]["strategy"] = "priority"
        path.write_text(yaml.safe_dump(document))
    changed = yaml.safe_load(source.read_text())
    changed["listeners"][0]["port"] = 8898
    source.write_text(yaml.safe_dump(changed))
    prepare()
    current = yaml.safe_load(path.read_text())
    assert current["routing"]["strategy"] == (
        "priority" if user_edited else "confidence"
    )
    assert current["listeners"][0]["port"] == (8899 if user_edited else 8898)
    assert_owned(path, not user_edited)
