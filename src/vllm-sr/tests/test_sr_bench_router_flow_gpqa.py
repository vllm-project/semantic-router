import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, catalog
from cli.sr_bench.datasets import DatasetReader
from cli.sr_bench.router_flow_gpqa import BENCHMARK, PROTOCOL
from cli.sr_bench.setup import PACKAGES, setup
from cli.sr_bench.sources import COUNTS


def test_gpqa_diagnostic_is_separate_from_published_benchmarks() -> None:
    adapter = get_adapter(BENCHMARK)
    assert adapter.version == PROTOCOL
    assert adapter.requires_answer is True
    assert adapter.prepare is None
    assert BENCHMARK not in BENCHMARK_WEIGHTS
    assert BENCHMARK not in COUNTS
    assert get_adapter("gpqa-diamond").version == "sr-bench-mcq-final-v2"
    assert BENCHMARK in {row["id"] for row in catalog()["benchmarks"]}


def test_missing_diagnostic_dataset_requires_local_preparation(tmp_path: Path) -> None:
    selected = DatasetReader(tmp_path).selection("smoke")
    row = next(row for row in selected["benchmarks"] if row["id"] == BENCHMARK)
    assert row["reason_code"] == "local_preparation_required"


def test_setup_inspects_the_optional_harness_without_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SR_BENCH_HOME", str(tmp_path))
    result = setup(benchmark=BENCHMARK)
    assert result["benchmarks"][0]["benchmark"] == BENCHMARK
    assert result["benchmarks"][0]["installed"] is False
    assert result["benchmarks"][0]["model_requests"] == 0
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    assert BENCHMARK not in {row["benchmark"] for row in setup()["benchmarks"]}
    assert PACKAGES[BENCHMARK]["version"] == "1.8.1"


def test_discovery_does_not_import_the_optional_framework(tmp_path: Path) -> None:
    code = """
import importlib.abc
import json
import sys
from importlib.metadata import EntryPoints
from cli.sr_bench import adapters

class NoEvalScope(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"evalscope", "bench"}:
            raise AssertionError("Catalog imported an optional dependency")

sys.meta_path.insert(0, NoEvalScope())
adapters.entry_points = lambda **kwargs: EntryPoints()
print(json.dumps([adapter.id for adapter in adapters.list_adapters()]))
"""
    shutil.copytree(
        Path(__file__).parents[1] / "cli",
        tmp_path / "cli",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert BENCHMARK in json.loads(result.stdout)
