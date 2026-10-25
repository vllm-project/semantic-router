from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.candidate_plans import candidate_manifest, validate_candidate_protocol
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, catalog
from cli.sr_bench.legacy_reasoning import CONFIGURATIONS
from cli.sr_bench.legacy_reasoning_explicit_cot import PROTOCOL as XC_PROTOCOL
from cli.sr_bench.legacy_reasoning_modes import (
    FIELD,
    canonical_mode,
    export_manifest,
    reasoning_controls,
)
from cli.sr_bench.recovery import recovery_plan
from cli.sr_bench.report import ComparisonValidator
from cli.sr_bench.store import Store, run_summary
from test_sr_bench_replay import record


def template(mode: str = "NR") -> dict[str, Any]:
    return {
        "version": "sr-bench-1.0",
        "cost_policy": "capability_only",
        "targets": [
            {
                "id": "off",
                "kind": "single",
                "model": "Qwen3-fixture",
                "base_url": "http://127.0.0.1:1/v1",
            }
        ],
        FIELD: {"version": 1, "conditions": [{"target_id": "off", "mode": mode}]},
    }


def test_explicit_cot_catalog_keeps_both_plain_protocols() -> None:
    entries = {item["id"]: item for item in catalog()["benchmarks"]}
    for name in (*CONFIGURATIONS, "mmlu-pro"):
        identity = f"legacy-{name}-explicit-cot"
        assert entries[identity]["adapter_version"] == XC_PROTOCOL
        assert identity not in BENCHMARK_WEIGHTS
    assert len(BENCHMARK_WEIGHTS) == 9
    assert get_adapter("legacy-mmlu-pro").version == "bench-mmlu-pro-plain-v1"
    assert get_adapter("legacy-arc").version == "bench-reasoning-plain-v1"
    assert "legacy-math-explicit-cot" not in entries
    assert "legacy-openmathreasoning-explicit-cot" not in entries


def test_mode_catalog_does_not_import_the_optional_runtime() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cli.sr_bench.contracts import catalog; catalog(); "
            "assert not {'reasoning','datasets','pandas','openai'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_native_mode_checks_require_explicit_collection(
    pytestconfig: pytest.Config,
) -> None:
    suite = Path(__file__).with_name("sr_bench_legacy_reasoning_modes_checks.py")
    assert suite.is_file()
    assert not any(
        suite.match(pattern) for pattern in pytestconfig.getini("python_files")
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("NR", "NR"),
        ("standard", "NR"),
        ("NR_REASONING", "NR_REASONING"),
        ("reasoning", "NR_REASONING"),
        ("AR", "AR"),
        ("XC", "XC"),
        ("Router_NR", "Router_NR"),
    ],
)
def test_requested_modes_have_explicit_aliases(requested: str, expected: str) -> None:
    assert canonical_mode(requested) == expected


@pytest.mark.parametrize("mode", ["cot", "ar", "", None, True, ["NR"]])
def test_unknown_modes_are_not_silently_plain(mode: Any) -> None:
    with pytest.raises(ValueError, match="reasoning mode"):
        canonical_mode(mode)


@pytest.mark.parametrize(
    "params",
    [
        {"reasoning": True},
        {"chat_template_kwargs": []},
        {"chat_template_kwargs": {"enable_thinking": "false"}},
        {"chat_template_kwargs": {"thinking": 0}},
        {"chat_template_kwargs": {"reasoning_effort": True}},
        {"chat_template_kwargs": {"reasoning_effort": "unlimited"}},
        {
            "chat_template_kwargs": {
                "thinking": False,
                "enable_thinking": True,
            }
        },
        {
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ],
)
def test_invalid_or_conflicting_controls_fail(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        reasoning_controls(params)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, {}),
        ({"reasoning_effort": "high"}, {"reasoning_effort": "high"}),
        (
            {"temperature": 0, "chat_template_kwargs": {"enable_thinking": False}},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            {"chat_template_kwargs": {"thinking": True, "date_string": "fixture"}},
            {"chat_template_kwargs": {"thinking": True}},
        ),
        (
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
            {"chat_template_kwargs": {"reasoning_effort": "low"}},
        ),
    ],
)
def test_controls_preserve_false_and_effort_levels(
    params: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert reasoning_controls(params) == expected


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"profile": "standard"}, "holdout"),
        ({"mode": "preview"}, "live"),
        ({"cases": []}, "template"),
        ({"execution_cells": []}, "template"),
        ({"sampling": {"reasoning_effort": "low"}}, "target"),
        (
            {
                FIELD: {
                    "version": 1,
                    "conditions": [{"target_id": "off", "mode": "cot"}],
                }
            },
            "reasoning mode",
        ),
        (
            {
                FIELD: {
                    "version": 1,
                    "conditions": [
                        {"target_id": "off", "mode": "NR"},
                        {"target_id": "off", "mode": "standard"},
                    ],
                }
            },
            "duplicate",
        ),
    ],
)
def test_invalid_compiler_controls_fail_before_source_or_runtime(
    updates: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        export_manifest(
            manifest={**template(), **updates},
            dataset="gsm8k",
            source_bundle=tmp_path / "missing.json",
        )


@pytest.mark.parametrize("seed", [True, -1, 2**32])
def test_mode_export_retains_native_seed_bounds(seed: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        export_manifest(
            manifest=template(),
            dataset="gsm8k",
            source_bundle=tmp_path / "missing.json",
            seed=seed,
        )


def test_saved_run_summary_preserves_condition_identity(tmp_path: Path) -> None:
    run = record(Store(tmp_path))
    contract = {"version": 1, "conditions": [], "selection_sha256": "fixture"}
    run["manifest"][FIELD] = contract
    assert run_summary(run)["manifest"][FIELD] == contract


def test_candidate_and_comparison_reject_condition_matrices(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run = record(store)
    run["manifest"][FIELD] = {"version": 1}
    with pytest.raises(ValueError, match="reasoning condition"):
        candidate_manifest(run, [])
    with pytest.raises(ValueError, match="reasoning condition"):
        ComparisonValidator(store, run)
    candidate = copy.deepcopy(run["manifest"])
    candidate.pop(FIELD)
    with pytest.raises(ValueError, match="reasoning condition"):
        validate_candidate_protocol(run, candidate)


def test_recovery_rejects_conditions_without_changing_saved_evidence(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    baseline = record(store)
    manifest = {**baseline["manifest"], FIELD: {"version": 1}}
    run, _ = store.create(manifest, "local")
    store.status(run["id"], "failed")
    before = json.dumps(store.get(run["id"]), sort_keys=True)
    with pytest.raises(ValueError, match="reasoning condition"):
        recovery_plan(store, run["id"])
    assert json.dumps(store.get(run["id"]), sort_keys=True) == before
