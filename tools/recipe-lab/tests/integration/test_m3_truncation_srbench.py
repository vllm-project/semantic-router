"""M3: a real sr-bench run with a tiny max_tokens reports truncation.

The mock backend finishes with ``length`` when the request caps output
below ``TRUNCATION_MAX_TOKENS`` tokens, so the sr-bench engine marks
every case ``quality_failure: "output_limit"`` and the target summary
carries ``output_diagnostics.output_limit_cases``. recipe-lab must turn
that per-case count (the W3 field) into the truncation rate; the
call-level finish-reason share is reference only. Skips unless a live
sr-bench service is present; never runs in CI.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from recipe_lab.live import arm_evidence_quality
from recipe_lab.splitting import load_cases
from recipe_lab.srbench import SrbenchCli, cases_to_srbench_rows

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            os.environ.get("RECIPE_LAB_SRBENCH_URL")
            and os.environ.get("RECIPE_LAB_SRBENCH_STORE")
        ),
        reason="RECIPE_LAB_SRBENCH_URL/RECIPE_LAB_SRBENCH_STORE are not set (M3)",
    ),
]

_TARGET_ID = "rl-single-trunc"
_BUCKETS = {"input": 0, "cached_input": 0, "cache_write": 0, "output": 0}


def _cases(smoke_cases: Path) -> list[dict]:
    """The smoke cases as sr-bench rows via the shared transform (X3).

    The answer is pinned to "A" (the answer the mock backend is graded
    against); the user message is the smoke case's prompt verbatim.
    """
    rows = cases_to_srbench_rows(load_cases(smoke_cases), benchmark="mmlu-pro")
    return [dict(row, answer="A") for row in rows]


def _single_target(mock_backend: dict) -> dict:
    return {
        "id": _TARGET_ID,
        "kind": "single",
        "model": "local/private-qwen",
        "base_url": mock_backend["endpoint"] + "/v1",
        "prices": {"local/private-qwen": _BUCKETS},
    }


def _register_targets(srbench_bin: str, srbench_store: str, registry: Path) -> None:
    completed = subprocess.run(
        [
            srbench_bin,
            "benchmark",
            "--store",
            srbench_store,
            "target",
            "register",
            "--file",
            str(registry),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_output_limit_cases_drive_the_truncation_rate(
    smoke_cases: Path,
    srbench_url: str,
    srbench_store: str,
    srbench_bin: str,
    mock_backend: dict,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    cases = _cases(smoke_cases)
    dataset.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    registry = tmp_path / "targets.json"
    registry.write_text(
        json.dumps({"targets": [_single_target(mock_backend)]}, indent=1) + "\n",
        encoding="utf-8",
    )
    _register_targets(srbench_bin, srbench_store, registry)

    # max_tokens caps the output below the mock's truncation threshold, so
    # every case finishes with "length" and becomes an output-limit case.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "sr-bench-1.0",
                "name": "recipe-lab-m3-truncation",
                "cost_policy": "capability_only",
                "targets": [_single_target(mock_backend)],
                "dataset": {
                    "path": str(dataset),
                    "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                },
                "sampling": {"max_tokens": 4},
                "limits": {
                    "total_timeout_s": 600,
                    "idle_timeout_s": 300,
                    "max_run_seconds": 1800,
                },
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    cli = SrbenchCli(bin=srbench_bin, url=srbench_url, store=srbench_store)
    run_doc = cli.run_manifest(manifest)
    run_id = run_doc["id"]
    assert isinstance(run_id, str) and run_id

    report = cli.report(run_id, tmp_path / "report.json")
    assert isinstance(report, dict)
    target = report["summary"]["targets"][0]
    diagnostics = target["output_diagnostics"]
    planned = diagnostics["planned_cases"]
    assert planned == len(cases)
    # every case hit the tiny token cap and was force-stopped (W3 field)
    assert diagnostics["output_limit_cases"] == planned
    assert diagnostics["result_cases"] == planned
    assert diagnostics["subject_calls"]["finish_reasons"].get("length") == planned

    # the per-case count is the truncation rate; the call share is reference
    evidence = arm_evidence_quality([target], routing_arm=False)
    assert evidence is not None
    assert evidence["truncation_rate"] == pytest.approx(1.0)
    assert evidence["missing_result_rate"] == 0.0
    assert evidence["no_answer_rate"] is None
    assert evidence["finish_reason_length_rate"] == pytest.approx(1.0)
