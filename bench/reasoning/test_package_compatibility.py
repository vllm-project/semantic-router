import json
import subprocess
import sys
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import pytest

from bench.reasoning.dataset_factory import DatasetFactory


def test_registered_reasoning_coverage_is_retained() -> None:
    assert set(DatasetFactory.get_available_datasets()) == {
        "mmlu",
        "mmlu-pro",
        "arc",
        "arc-easy",
        "arc-challenge",
        "gpqa",
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
    }
    assert len(set(DatasetFactory._registered_datasets.values())) == 16


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("mmlu", "mmlu-pro"), ("gpqa", "gpqa-main")],
)
def test_dataset_aliases_keep_the_same_implementation(
    alias: str, canonical: str
) -> None:
    assert type(DatasetFactory.create_dataset(alias)) is type(
        DatasetFactory.create_dataset(canonical)
    )


def test_installed_package_keeps_reasoning_and_looper_entrypoints() -> None:
    package = distribution("vllm-semantic-router-bench")
    assert package.version == "1.1.0"
    assert package.metadata["Requires-Python"] == ">=3.8"
    assert {
        entry.name: entry.value
        for entry in package.entry_points
        if entry.group == "console_scripts"
    } == {
        "vllm-semantic-router-bench": "reasoning.cli:main",
        "router-bench": "reasoning.router_reason_bench_multi_dataset:main",
        "bench-plot": "reasoning.bench_plot:main",
        "reasoning-mode-eval": "reasoning.reasoning_mode_eval:main",
        "looper-tts": "looper_tts.__main__:main",
    }
    assert not any(
        requirement.startswith("vllm-sr") for requirement in package.requires
    )


def test_installed_looper_uses_its_packaged_fixture(tmp_path: Path) -> None:
    fixture = files("looper_tts").joinpath("testdata", "synthetic.json")
    result = subprocess.run(
        [sys.executable, "-m", "looper_tts", "validate", "--config", str(fixture)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["valid"] is True
    output = tmp_path / "plan"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "looper_tts",
            "plan",
            "--config",
            str(fixture),
            "--code-revision",
            "package-contract",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["cells"] > 0
    assert (output / "manifest.json").is_file()
