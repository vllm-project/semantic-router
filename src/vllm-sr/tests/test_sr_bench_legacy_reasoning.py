from __future__ import annotations

import os
import subprocess
import sys

from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, BENCHMARKS, catalog
from cli.sr_bench.setup import setup

CONFIGURATIONS: tuple[str, ...] = (
    "arc",
    "arc-easy",
    "arc-challenge",
    "gpqa-main",
    "gpqa-extended",
    "gpqa-diamond",
    "gsm8k",
    "aqua-rat",
    "drop",
    "strategyqa",
    "sciq",
    "openbookqa",
    "truthfulqa",
    "commonsenseqa",
    "hellaswag",
)


def test_legacy_catalog_registers_distinct_opt_in_diagnostics() -> None:
    published = {row[0] for row in BENCHMARKS}
    entries = {row["id"]: row for row in catalog()["benchmarks"]}
    for configuration in CONFIGURATIONS:
        identity = "legacy-" + configuration
        adapter = get_adapter(identity)
        assert adapter.requires_answer
        assert adapter.version == "bench-reasoning-plain-v1"
        assert "legacy plain diagnostic" in entries[identity]["title"]
        assert identity not in published
        assert identity not in BENCHMARK_WEIGHTS
    assert "legacy-math" not in entries
    assert "legacy-openmathreasoning" not in entries


def test_catalog_does_not_import_native_library() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cli.sr_bench.contracts import catalog; catalog(); "
            "assert 'reasoning' not in sys.modules; "
            "assert 'datasets' not in sys.modules; "
            "assert 'pandas' not in sys.modules; "
            "assert 'openai' not in sys.modules",
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_default_setup_does_not_include_legacy_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SR_BENCH_HOME", str(tmp_path))
    names = {row["benchmark"] for row in setup(install=False)["benchmarks"]}
    assert not any(name.startswith("legacy-") for name in names)
