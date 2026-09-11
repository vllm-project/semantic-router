from __future__ import annotations

import json
import stat
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from cli.commands.benchmark import benchmark
from cli.evaluation import intelligence_benchmarks
from cli.evaluation.intelligence_benchmarks import (
    IntelligenceRunOptions,
    build_intelligence_run_plan,
    normalize_openai_base_url,
    run_intelligence_benchmarks,
    select_intelligence_benchmarks,
)
from click.testing import CliRunner


def _options(tmp_path: Path, **updates: object) -> IntelligenceRunOptions:
    values: dict[str, object] = {
        "model": "vllm-sr/quality",
        "base_url": "http://127.0.0.1:8801",
        "source_root": tmp_path / "sources",
        "output_root": tmp_path / "results",
    }
    values.update(updates)
    return IntelligenceRunOptions(**values)  # type: ignore[arg-type]


def test_intelligence_catalog_locks_hle_to_text_only() -> None:
    result = CliRunner().invoke(benchmark, ["intelligence", "list"])

    assert result.exit_code == 0, result.output
    catalog = json.loads(result.output)
    assert len(catalog["benchmarks"]) == 6
    hle = next(
        item
        for item in catalog["benchmarks"]
        if item["id"] == "cais/humanitys-last-exam@1.0.0"
    )
    assert hle["profile"] == "independent-text-only"
    assert hle["sample_count"] == 2158
    assert hle["data"][0]["source"].endswith(":text-only-2158")
    assert hle["required_environment"] == ["HF_TOKEN", "OPENROUTER_API_KEY"]


def test_hle_plan_forces_text_only_and_contains_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "subject-secret")
    hle = select_intelligence_benchmarks(("cais/humanitys-last-exam@1.0.0",))[0]

    plan = build_intelligence_run_plan(hle, _options(tmp_path))

    command = plan["execution"]["command"]
    assert "include_multi_modal=false" in command
    assert "subject-secret" not in json.dumps(plan)
    assert plan["index_eligible_protocol"] is True


def test_terminal_plan_pins_dataset_and_marks_smoke_run_ineligible(
    tmp_path: Path,
) -> None:
    terminal = select_intelligence_benchmarks(("harbor/terminal-bench@2.1.0",))[0]

    plan = build_intelligence_run_plan(
        terminal,
        _options(tmp_path, sample_limit=2, terminal_attempts=1),
    )

    command = plan["execution"]["command"]
    dataset = command[command.index("--dataset") + 1]
    assert dataset.endswith(
        "@sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
    )
    assert plan["index_eligible_protocol"] is False


def test_intelligence_run_writes_private_receipts_without_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_definition = select_intelligence_benchmarks(
        ("tiger-ai-lab/mmlu-pro@1.0.0",)
    )[0]
    monkeypatch.setenv("ROUTER_BENCHMARK_TOKEN", "never-write-this")
    monkeypatch.setattr(
        intelligence_benchmarks.shutil, "which", lambda _name: "/usr/bin/uv"
    )
    monkeypatch.setattr(
        intelligence_benchmarks,
        "_verify_source_checkout",
        lambda _source, _path: None,
    )
    monkeypatch.setattr(
        intelligence_benchmarks,
        "_verify_data_pins",
        lambda _data: None,
    )
    monkeypatch.setattr(
        intelligence_benchmarks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    options = _options(tmp_path, api_key_env="ROUTER_BENCHMARK_TOKEN")

    result = run_intelligence_benchmarks((benchmark_definition,), options)

    assert result["completed"] is True
    receipt = options.output_root / "suite-receipt.json"
    assert receipt.is_file()
    assert "never-write-this" not in receipt.read_text(encoding="utf-8")
    if stat.S_IMODE(receipt.stat().st_mode) != 0o600:
        pytest.skip("temporary filesystem does not preserve POSIX private modes")


def test_runner_environment_exposes_only_required_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("ROUTER_BENCHMARK_TOKEN", "model-token")
    monkeypatch.setenv("HF_TOKEN", "dataset-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "judge-token")
    monkeypatch.setenv("VSR_MGMT_TOKEN", "management-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cloud-secret")

    environment = intelligence_benchmarks._runner_environment(
        "ROUTER_BENCHMARK_TOKEN",
        ["HF_TOKEN", "OPENROUTER_API_KEY"],
    )

    assert environment["PATH"] == "/usr/bin"
    assert environment["OPENAI_API_KEY"] == "model-token"
    assert environment["HF_TOKEN"] == "dataset-token"
    assert environment["OPENROUTER_API_KEY"] == "judge-token"
    assert "ROUTER_BENCHMARK_TOKEN" not in environment
    assert "VSR_MGMT_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://localhost:8801", "http://localhost:8801/v1"),
        ("https://router.example/prefix/v1/", "https://router.example/prefix/v1"),
    ],
)
def test_normalize_openai_base_url(value: str, expected: str) -> None:
    assert normalize_openai_base_url(value) == expected


def test_normalize_openai_base_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="api-key-env"):
        normalize_openai_base_url("https://user:secret@router.example")


def test_intelligence_plan_rejects_noncanonical_api_key_environment(
    tmp_path: Path,
) -> None:
    benchmark_definition = select_intelligence_benchmarks(
        ("tiger-ai-lab/mmlu-pro@1.0.0",)
    )[0]

    with pytest.raises(ValueError, match="uppercase environment variable"):
        build_intelligence_run_plan(
            benchmark_definition,
            _options(tmp_path, api_key_env="aws.secret"),
        )


def test_dataset_revision_attestation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_definition = select_intelligence_benchmarks(
        ("tiger-ai-lab/mmlu-pro@1.0.0",)
    )[0]
    monkeypatch.setattr(
        intelligence_benchmarks,
        "_hugging_face_dataset_head",
        lambda _repository: "new-revision",
    )

    with pytest.raises(ValueError, match="moved from frozen revision"):
        intelligence_benchmarks._verify_data_pins(
            [asdict(pin) for pin in benchmark_definition.data]
        )


def test_scicode_uses_pinned_inspect_assets_without_manual_input(
    tmp_path: Path,
) -> None:
    scicode = select_intelligence_benchmarks(("scicode-bench/scicode@1.0.0",))[0]

    plan = build_intelligence_run_plan(scicode, _options(tmp_path))

    command = plan["execution"]["command"]
    assert "inspect_evals/scicode" in command
    assert "provide_scientific_background=false" in command
    assert "include_dev_set=false" in command
    revisions = {pin["revision"] for pin in plan["benchmark"]["data"]}
    assert any("48b0272" in revision for revision in revisions)
