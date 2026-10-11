"""The Vela 2.0 bench tool's model loads (``tools/vela2_bench.py``)."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "vela2_bench.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("vela2_bench", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_path_switches_reach_the_max_speed_load(bench, monkeypatch) -> None:
    loads = []

    def load_runtime(package, device, engine="native", reduced=None, options=None):
        loads.append((reduced, options))
        if reduced is not None:
            raise StopIteration
        return object()

    monkeypatch.setattr(bench, "load_runtime", load_runtime)
    monkeypatch.setattr(bench, "pin_choices", lambda package, device: None)
    monkeypatch.setattr(bench, "device_executor", lambda device: lambda work: work())
    monkeypatch.setattr(
        sys,
        "argv",
        ["vela2_bench.py", "--package", "pkg", "--device", "cpu", "--output", "out.json",
         "--sides", "max_speed:bfloat16", "--no-graphs", "--no-fused"],
    )  # fmt: skip
    with pytest.raises(StopIteration):
        bench.main()

    assert [reduced for reduced, _ in loads] == [None, "bfloat16"]
    for _, options in loads:
        assert not options.graphs and not options.fused_kernels
