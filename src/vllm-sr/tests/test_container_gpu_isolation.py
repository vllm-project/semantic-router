from cli.container_gpu_isolation import (
    AMD_ROUTER_VISIBLE_DEVICES_ENV,
    router_runtime_env,
)


def test_router_runtime_env_isolates_amd_router(monkeypatch):
    monkeypatch.setenv(AMD_ROUTER_VISIBLE_DEVICES_ENV, "7")

    common = {"VLLM_SR_PLATFORM": "amd"}
    result = router_runtime_env(common, "amd")

    assert result["ROCR_VISIBLE_DEVICES"] == "7"
    assert "ROCR_VISIBLE_DEVICES" not in common


def test_router_runtime_env_ignores_isolation_for_other_platforms(monkeypatch):
    monkeypatch.setenv(AMD_ROUTER_VISIBLE_DEVICES_ENV, "7")

    result = router_runtime_env({"VLLM_SR_PLATFORM": "nvidia"}, "nvidia")

    assert "ROCR_VISIBLE_DEVICES" not in result


def test_router_runtime_env_ignores_blank_isolation(monkeypatch):
    monkeypatch.setenv(AMD_ROUTER_VISIBLE_DEVICES_ENV, "  ")

    result = router_runtime_env({"VLLM_SR_PLATFORM": "amd"}, "amd")

    assert "ROCR_VISIBLE_DEVICES" not in result
