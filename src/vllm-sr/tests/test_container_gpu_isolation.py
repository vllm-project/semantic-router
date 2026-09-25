from pathlib import Path
from types import SimpleNamespace

import pytest
from cli import container_gpu_isolation, container_start
from cli.container_gpu_isolation import (
    AMD_ROUTER_VISIBLE_DEVICES_ENV,
    COMGR_CACHE_CONTAINER_PATH,
    router_compiler_cache,
    router_runtime_env,
)
from cli.runtime_stack import resolve_runtime_stack

IMAGE_A = "sha256:" + "a" * 64
IMAGE_B = "sha256:" + "b" * 64


def test_amd_compiler_cache_reuses_only_the_same_stack_and_image(tmp_path, monkeypatch):
    inspected_image = IMAGE_A

    def inspect(command, **kwargs):
        assert command == [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "router:latest",
        ]
        assert kwargs["timeout"] == 10
        return SimpleNamespace(returncode=0, stdout=inspected_image + "\n")

    monkeypatch.setattr(container_gpu_isolation.subprocess, "run", inspect)
    state = tmp_path / ".vllm-sr"
    state.mkdir(mode=0o700)

    first = router_compiler_cache(
        "docker", "router:latest", str(state), "stack-a", "amd"
    )
    assert first is not None
    assert first.image_id == IMAGE_A
    assert first.mount.endswith(f":{COMGR_CACHE_CONTAINER_PATH}:z")
    cache_path = Path(first.mount.split(":", 1)[0])
    assert cache_path.is_dir()
    assert cache_path.stat().st_mode & 0o777 == 0o700
    assert cache_path.parent.stat().st_mode & 0o777 == 0o700
    cache_path.joinpath("warmup").write_text("cached")

    again = router_compiler_cache(
        "docker", "router:latest", str(state), "stack-a", "amd"
    )
    assert again == first
    assert cache_path.joinpath("warmup").read_text() == "cached"

    other_stack = router_compiler_cache(
        "docker", "router:latest", str(state), "stack-b", "amd"
    )
    assert other_stack is not None and other_stack.mount != first.mount

    inspected_image = IMAGE_B
    other_image = router_compiler_cache(
        "docker", "router:latest", str(state), "stack-a", "amd"
    )
    assert other_image is not None and other_image.mount != first.mount
    assert other_image.image_id == IMAGE_B


def test_amd_compiler_cache_skips_uninspectable_image_and_other_platforms(
    tmp_path, monkeypatch
):
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(container_gpu_isolation.subprocess, "run", inspect)
    state = tmp_path / ".vllm-sr"
    assert (
        router_compiler_cache("docker", "router:latest", str(state), "a", "cpu") is None
    )
    assert calls == []
    assert (
        router_compiler_cache("docker", "router:latest", str(state), "a", "amd") is None
    )
    assert len(calls) == 1
    assert not state.exists()


def test_amd_compiler_cache_rejects_symlinked_private_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        container_gpu_isolation.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=IMAGE_A),
    )
    state = tmp_path / ".vllm-sr"
    state.mkdir(mode=0o700)
    (state / "compiler-cache").symlink_to(tmp_path)

    with pytest.raises(ValueError, match="symbolic link"):
        router_compiler_cache("docker", "router:latest", str(state), "a", "amd")


def test_router_command_pins_amd_image_and_mounts_its_compiler_cache(
    tmp_path, monkeypatch
):
    cache = container_gpu_isolation.AMDCompilerCache(
        image_id=IMAGE_A,
        mount=f"{tmp_path}/cache:{COMGR_CACHE_CONTAINER_PATH}:z",
    )
    monkeypatch.setattr(container_start, "router_compiler_cache", lambda *args: cache)
    monkeypatch.setenv("VLLM_SR_AMD_GPU_PASSTHROUGH", "0")
    paths = {
        "vllm_sr_dir": str(tmp_path / ".vllm-sr"),
        "source_config_path": str(tmp_path / "config.yaml"),
        "log_spool_logs_root": str(tmp_path / "logs"),
        "models_dir": str(tmp_path / "models"),
        "log_spool_router_mount": (
            f"{tmp_path}/router.log:/var/log/vllm-sr-producer/current.log:z"
        ),
        "runtime_container_config": "/app/.vllm-sr/runtime-config.yaml",
    }
    command = container_start._build_router_runtime_command(
        runtime="docker",
        router_image="router:latest",
        nofile_limit=4096,
        runtime_network_name="test-network",
        normalized_platform="amd",
        common_env={},
        runtime_paths=paths,
        stack_layout=resolve_runtime_stack(stack_name="cache-test"),
        inherited_sensitive_env=set(),
        management_listener={"host_port": 8080, "port": 8080},
    )

    assert cache.mount in command
    assert IMAGE_A in command
    assert "router:latest" not in command


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
