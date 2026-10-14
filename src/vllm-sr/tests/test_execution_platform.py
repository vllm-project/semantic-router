"""Platform discovery, visible-device mapping and target isolation."""

import json

import pytest
from cli import execution_platform as platform
from cli.config_translator import _apply_gateway_and_platform

read_cluster_devices = platform._cluster_devices


@pytest.mark.parametrize("value", ["cpu", "cuda", "rocm"])
def test_explicit_platform_does_not_probe_an_unrelated_host(monkeypatch, value):
    monkeypatch.setattr(
        platform, "_host_devices", lambda: pytest.fail("unexpected discovery")
    )
    assert platform.resolve_execution_platform(value, target="kubernetes") == value


@pytest.mark.parametrize("value", ["amd", "nvidia", "CUDA", "unknown"])
def test_platform_names_have_no_vendor_aliases(value):
    with pytest.raises(ValueError, match="auto, cpu, cuda or rocm"):
        platform.resolve_execution_platform(value)


def test_auto_uses_target_cluster_not_control_host(monkeypatch):
    monkeypatch.setattr(
        platform, "_host_devices", lambda: pytest.fail("control host inspected")
    )
    contexts = []
    monkeypatch.setattr(
        platform,
        "_cluster_devices",
        lambda context: contexts.append(context) or {"rocm": (0, 1), "cuda": ()},
    )
    assert (
        platform.resolve_execution_platform(
            "auto", target="kubernetes", context="my-cluster"
        )
        == "rocm"
    )
    assert contexts == ["my-cluster"]


def test_auto_requires_disambiguation_for_mixed_accelerators(monkeypatch):
    monkeypatch.setattr(platform, "_host_devices", lambda: {"rocm": (0,), "cuda": (0,)})
    with pytest.raises(ValueError, match="Both CUDA and ROCm"):
        platform.resolve_execution_platform("auto")


def test_auto_honors_visibility_before_selecting_backend(monkeypatch):
    monkeypatch.setattr(platform, "_host_devices", lambda: {"rocm": (0,), "cuda": (0,)})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert platform.resolve_execution_platform("auto") == "rocm"


def test_physical_gpu_ids_map_through_effective_rocm_mask(monkeypatch):
    monkeypatch.setenv("VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES", "7,3")
    monkeypatch.setenv(
        "ROCR_VISIBLE_DEVICES", "1"
    )  # Container isolation overrides this.
    visible = platform.visible_device_ids("rocm", tuple(range(8)))
    assert visible == (7, 3)
    assert platform.device_ordinals((3, 7), "rocm", visible) == (1, 0)
    assert platform.runtime_gpu_environment("rocm") == {"ROCR_VISIBLE_DEVICES": "7,3"}
    with pytest.raises(ValueError, match="outside"):
        platform.device_ordinals((0,), "rocm", visible)


def test_unmasked_ids_remain_physical_ordinals():
    assert platform.device_ordinals((7, 3), "rocm", tuple(range(8))) == (7, 3)


def test_cuda_mask_is_forwarded_and_validated(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,2")
    assert platform.visible_device_ids("cuda", tuple(range(6))) == (5, 2)
    assert platform.device_ordinals((2,), "cuda", (5, 2)) == (1,)
    assert platform.runtime_gpu_environment("cuda") == {"CUDA_VISIBLE_DEVICES": "5,2"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    with pytest.raises(ValueError, match="unavailable"):
        platform.visible_device_ids("cuda", tuple(range(6)))


def test_cluster_discovery_uses_ready_schedulable_nodes(monkeypatch):
    def node(count, *, ready=True, unschedulable=False):
        return {
            "spec": {"unschedulable": unschedulable},
            "status": {
                "allocatable": {"amd.com/gpu": str(count)},
                "conditions": [
                    {"type": "Ready", "status": "True" if ready else "False"}
                ],
            },
        }

    commands = []
    monkeypatch.setattr(
        platform,
        "_output",
        lambda args: commands.append(args)
        or json.dumps(
            {"items": [node(8, ready=False), node(4, unschedulable=True), node(2)]}
        ),
    )
    assert read_cluster_devices("chosen") == {"cuda": (), "rocm": (0, 1)}
    assert commands == [
        ["kubectl", "--context", "chosen", "get", "nodes", "-o", "json"]
    ]


@pytest.mark.parametrize(
    "devices,count", [(["rocm:0"] * 4, 1), (["rocm:0", "rocm:3"], 4)]
)
def test_kubernetes_allocates_devices_not_process_count(devices, count):
    document = {
        "global": {
            "model_catalog": {
                "deployments": {
                    "primary": {
                        "provider": "model_runtime",
                        "artifact": "acme/model",
                        "replicas": [{"device": value} for value in devices],
                    }
                }
            }
        }
    }
    values = {}
    _apply_gateway_and_platform(
        values,
        gateway="standalone",
        platform="rocm",
        image_overridden=False,
        config=document,
    )
    assert values["resources"]["limits"]["amd.com/gpu"] == count
    assert values["resources"]["requests"]["amd.com/gpu"] == count
    assert "rocm" in values["image"]["repository"]
