from importlib import metadata

import pytest
from vllm_sr_runtime.accel.kernels import Kernel, KernelSet, reference_kernels
from vllm_sr_runtime.plugins import registry
from vllm_sr_runtime.plugins.base import (
    Accelerator,
    DtypePolicy,
    Engine,
    EngineOptions,
    ModelFamily,
    Profile,
)


@pytest.fixture(autouse=True)
def fresh_discovery():
    registry.discover.cache_clear()
    yield
    registry.discover.cache_clear()


def test_builtin_plugins_are_discoverable():
    found = registry.discover()
    assert {"decision2"} <= set(found["families"])
    assert {"native"} <= set(found["engines"])
    assert {"cpu", "cuda", "rocm"} <= set(found["accelerators"])
    assert {"exact", "shared_context", "batching", "max_speed"} <= set(
        found["profiles"]
    )
    assert issubclass(registry.plugin("families", "decision2").load(), ModelFamily)
    assert issubclass(registry.plugin("engines", "native").load(), Engine)
    assert issubclass(registry.plugin("accelerators", "rocm").load(), Accelerator)
    assert issubclass(registry.plugin("profiles", "exact").load(), Profile)


def test_profiles_declare_their_numerics():
    numerics = {
        name: registry.plugin("profiles", name).load().numerics
        for name in registry.names("profiles")
    }
    assert numerics["exact"] == "exact"
    assert {numerics[name] for name in ("shared_context", "batching", "max_speed")} == {
        "approximate"
    }


def test_only_max_speed_asks_for_reduced_precision():
    base = EngineOptions()
    asks = {
        name: registry.instantiate("profiles", name)
        .engine_options(base)
        .reduced_precision
        for name in ("exact", "shared_context", "batching", "max_speed")
    }
    assert asks == {
        "exact": False,
        "shared_context": False,
        "batching": False,
        "max_speed": True,
    }
    assert DtypePolicy().reduced_gpu is None and DtypePolicy().reduced_cpu is None


def test_third_party_entry_points_are_loaded(monkeypatch):
    class FakeEntry:
        def __init__(self, name, value):
            self.name, self.value, self.dist = (
                name,
                value,
                type("D", (), {"name": "acme", "version": "1.0"})(),
            )

    real = metadata.entry_points

    def entry_points(group):
        extra = [
            FakeEntry("acme_label_token", "vllm_sr_runtime.profiles.exact:ExactProfile")
        ]
        return list(real(group=group)) + (
            extra if group == "vllm_sr_runtime.families" else []
        )

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    entry = registry.discover()["families"]["acme_label_token"]
    assert (
        entry.distribution == "acme"
        and entry.describe()["group"] == "vllm_sr_runtime.families"
    )


def test_conflicting_names_are_refused(monkeypatch):
    class FakeEntry:
        def __init__(self, value, dist):
            self.name, self.value, self.dist = (
                "decision2",
                value,
                type("D", (), {"name": dist, "version": "1"})(),
            )

    def entry_points(group):
        if group == "vllm_sr_runtime.families":
            return [FakeEntry("a:B", "one"), FakeEntry("c:D", "two")]
        return []

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    with pytest.raises(registry.PluginConflictError):
        registry.discover()


def test_unknown_plugin_lists_alternatives():
    with pytest.raises(KeyError, match="available"):
        registry.plugin("engines", "vllm")


def test_kernel_registry_prefers_exact_kernels():
    kernels = reference_kernels("cpu")
    assert kernels.select("sdpa").source == "torch-reference"
    kernels.register(
        Kernel("sdpa", lambda *a, **k: None, "fast-but-approximate", exact=False)
    )
    assert kernels.select("sdpa").source == "torch-reference"
    kernels.allow_approximate = True
    assert kernels.select("sdpa").source == "fast-but-approximate"
    with pytest.raises(KeyError):
        KernelSet("cpu").select("missing")
