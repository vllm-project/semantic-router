"""Storage provisioning coverage for split runtime startup."""

from cli import core, runtime_lifecycle


def _backend_provisioning_config():
    return {
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "services": {
                "response_api": {"store_backend": "redis"},
                "router_replay": {"store_backend": "postgres"},
            }
        },
    }


def test_start_vllm_sr_loads_runtime_config_for_backend_provisioning(monkeypatch):
    load_paths = []
    provisioned = {}

    def record(name, ret=(0, "", "")):
        def _fn(*args, **kwargs):
            provisioned.setdefault("calls", []).append((name, args, kwargs))
            return ret

        return _fn

    def fake_load_config(path):
        load_paths.append(path)
        return _backend_provisioning_config()

    monkeypatch.setattr(core, "print_vllm_logo", lambda: None)
    monkeypatch.setattr(core, "ensure_clean_runtime_container", lambda _name: None)
    monkeypatch.setattr(core, "load_config", fake_load_config)
    monkeypatch.setattr(
        core,
        "provision_storage_backends",
        lambda config, stack_layout, **_kwargs: (
            provisioned.update(
                {
                    "config": config,
                    "network_name": stack_layout.data_network_name,
                    "stack_layout": stack_layout,
                }
            )
            or {"milvus", "redis", "postgres"}
        ),
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_status",
        lambda _name: "running",
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_create_network",
        record("container_create_network"),
    )
    monkeypatch.setattr(
        core, "container_start_vllm_sr", record("container_start_vllm_sr")
    )
    monkeypatch.setattr(
        runtime_lifecycle,
        "container_network_connect",
        record("container_network_connect"),
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs_since", lambda *args, **kwargs: (0, "", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_exec", lambda *args, **kwargs: (0, "ok", "")
    )
    monkeypatch.setattr(
        runtime_lifecycle, "load_openclaw_registry", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        runtime_lifecycle, "container_logs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(core, "_wait_and_verify_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        core, "recover_openclaw_containers", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(core, "log_runtime_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "maybe_finish_setup_mode", lambda *args, **kwargs: False)

    core.start_vllm_sr(
        "/tmp/effective-config.yaml",
        env_vars={},
        enable_observability=False,
        source_config_file="/tmp/source-config.yaml",
        runtime_config_file="/tmp/runtime-config.yaml",
    )

    assert load_paths == ["/tmp/runtime-config.yaml"]
    assert provisioned["config"]["global"]["services"]["response_api"][
        "store_backend"
    ] == ("redis")
    assert (
        provisioned["config"]["global"]["services"]["router_replay"]["store_backend"]
        == "postgres"
    )
    runtime_start = next(
        call for call in provisioned["calls"] if call[0] == "container_start_vllm_sr"
    )
    assert runtime_start[2]["env_vars"]["VLLM_SR_MANAGED_STORAGE_BACKENDS"] == (
        "milvus,postgres,redis"
    )
