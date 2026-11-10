from importlib import metadata

import pytest
from vllm_srun.accel.kernels import Kernel, KernelSet, reference_kernels
from vllm_srun.plugins import registry
from vllm_srun.plugins.base import (
    Accelerator,
    DtypePolicy,
    Engine,
    ModelFamily,
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
    assert set(found) == {"families", "engines", "accelerators"}
    assert issubclass(registry.plugin("families", "decision2").load(), ModelFamily)
    assert issubclass(registry.plugin("engines", "native").load(), Engine)
    assert issubclass(registry.plugin("accelerators", "rocm").load(), Accelerator)


def test_every_plugin_loads_under_its_own_name():
    for kind, entries in registry.discover().items():
        for name, entry in entries.items():
            assert entry.load().name == name, (kind, name)


def test_engine_and_accelerator_cards_list_where_auto_tries_them():
    """``/v1/models`` lists the priority ``auto`` reads, so a card shows why ``auto`` chose a plugin."""
    for kind in ("engines", "accelerators"):
        for name, entry in registry.discover()[kind].items():
            listed = entry.capabilities()["auto_priority"]
            assert listed == entry.load().auto_priority, (kind, name)
    engines = registry.discover()["engines"]
    assert engines["native"].capabilities()["auto_priority"] == 0
    assert engines["onnxruntime"].capabilities()["auto_priority"] is None


def test_an_entry_point_must_name_a_plugin_of_its_group():
    engine = registry.PluginEntry(
        "families",
        "native",
        "vllm_srun.engines.native.engine:NativeEngine",
        "acme",
        "1",
    )
    with pytest.raises(registry.PluginError, match="not a ModelFamily"):
        engine.load()
    renamed = registry.PluginEntry(
        "engines", "turbo", "vllm_srun.engines.native.engine:NativeEngine", "acme", "1"
    )
    with pytest.raises(registry.PluginError, match="whose name is 'native'"):
        renamed.load()


def test_engine_cards_list_the_dtypes_they_run():
    engines = registry.discover()["engines"]
    assert engines["native"].capabilities()["dtypes"] == [
        "auto",
        "float32",
        "bfloat16",
        "float16",
    ]
    assert engines["onnxruntime"].capabilities()["dtypes"] == ["auto"]


def test_families_consent_to_copies_and_approximate_kernels_explicitly():
    policy = DtypePolicy()
    assert policy.reduced_gpu is None and policy.reduced_cpu is None
    assert not policy.approximate_kernels


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
        extra = [FakeEntry("acme_label_token", "acme.family:LabelTokenFamily")]
        return list(real(group=group)) + (
            extra if group == "vllm_srun.families" else []
        )

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    entry = registry.discover()["families"]["acme_label_token"]
    assert (
        entry.distribution == "acme"
        and entry.describe()["group"] == "vllm_srun.families"
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
        if group == "vllm_srun.families":
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
