"""R1: threshold-candidate hot reload against the real router (V2).

Criterion (V2-1): every threshold-class candidate switches into the live
router through the management API (``reload=hot``, never restart-fallback)
and the switch-time median stays within 5 seconds. The identical candidate
set must also rank identically under the restart runner, so a hot-reload
regression cannot hide behind different routing outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

import pytest
from recipe_lab.cli import main

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RECIPE_LAB_ROUTER_BIN"),
        reason="RECIPE_LAB_ROUTER_BIN is not set; real router required (V2)",
    ),
]


def _run_cli(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict:
    """Invoke the CLI in-process and parse its stdout JSON document.

    ``_print_json`` pretty-prints with ``indent=2``, so the document spans
    several lines and must be parsed as a whole, not per line.
    """
    assert main(argv) == 0
    out = capsys.readouterr().out
    try:
        document = json.loads(out)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"no JSON document on stdout for: {' '.join(argv)}"
        ) from exc
    assert isinstance(document, dict), document
    return document


def _notes_statuses(notes: str) -> list:
    """The ``statuses=<json>`` prefix of the run ledger notes."""
    head = notes.split(" reload=", 1)[0]
    assert head.startswith("statuses="), notes
    return json.loads(head.removeprefix("statuses="))


def _notes_reload(notes: str) -> str:
    """The ``reload=<mode>`` token of the run ledger notes."""
    return notes.split(" reload=", 1)[1].split(" ", 1)[0]


def test_threshold_candidates_hot_reload_matches_restart(
    router_bin: str,
    smoke_dir: Path,
    mock_backend: dict,
    backend_port: int,
    hf_home: str | None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = smoke_dir / "cases.jsonl"

    # Render the spec: absolute manifest path plus the frozen content hash
    # of exactly that file (A4).
    cases_hash = "sha256:" + hashlib.sha256(cases.read_bytes()).hexdigest()
    spec_text = (
        (smoke_dir / "smoke-spec.yaml.tmpl")
        .read_text(encoding="utf-8")
        .replace("${CASES_FILE}", str(cases))
        .replace("${CASES_HASH}", cases_hash)
    )
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(spec_text, encoding="utf-8")

    # Baseline config: repoint every backend_ref at this test's own mock
    # backend port so the runner-managed instances cannot collide.
    config_text = (
        (smoke_dir / "smoke-config.yaml")
        .read_text(encoding="utf-8")
        .replace("127.0.0.1:8000", f"127.0.0.1:{backend_port}")
    )
    base_config = tmp_path / "smoke-config.yaml"
    base_config.write_text(config_text, encoding="utf-8")

    # Answer bank against the shared mock backend.
    bank = tmp_path / "bank.jsonl"
    _run_cli(
        capsys,
        [
            "bank",
            "build",
            "--cases",
            str(cases),
            "--models",
            "local/private-qwen,cloud/frontier-reasoning",
            "--endpoint",
            mock_backend["endpoint"],
            "--output",
            str(bank),
        ],
    )

    # Threshold candidates: the zero delta is dropped and the values clamp
    # to [0, 1], so deltas -0.2/0/0.2/0.35 on the 0.6 depth threshold yield
    # exactly three candidates (0.4 / 0.8 / 0.95).
    candidates_dir = tmp_path / "candidates"
    proposal = _run_cli(
        capsys,
        [
            "propose-thresholds",
            "--base-config",
            str(base_config),
            "--spec",
            str(spec_path),
            "--signal",
            "depth",
            # `=` form: argparse would read a leading-dash value as an option.
            "--deltas=-0.2,0,0.2,0.35",
            "--output-dir",
            str(candidates_dir),
        ],
    )
    assert len(proposal["candidates"]) == 3

    ledgers: dict[str, list[dict]] = {}
    run_outputs: dict[str, dict] = {}
    for mode in ("restart", "hot"):
        ledger_path = tmp_path / f"ledger-{mode}.jsonl"
        output_dir = tmp_path / f"run-{mode}"
        output_dir.mkdir()
        argv = [
            "run",
            "--spec",
            str(spec_path),
            "--cases",
            str(cases),
            "--bank",
            str(bank),
            "--candidates",
            str(candidates_dir),
            "--baseline-config",
            str(base_config),
            "--proposer",
            "threshold-search",
            "--ledger",
            str(ledger_path),
            "--output-dir",
            str(output_dir),
            "--router-bin",
            router_bin,
            "--mock-backend",
            str(smoke_dir / "mock_backend.py"),
            "--mock-extra-args",
            f"--cases {cases}",
            "--backend-port",
            str(backend_port),
            "--reload",
            mode,
        ]
        if hf_home:
            argv += ["--hf-home", hf_home]
        run_outputs[mode] = _run_cli(capsys, argv)
        ledgers[mode] = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    for mode in ("restart", "hot"):
        outcome = run_outputs[mode]
        assert outcome["evaluated"] == 3
        assert outcome["timed_out"] == []
        assert outcome["rejected"] == []
        ranked = [entry for entry in outcome["results"] if entry["status"] == "ranked"]
        assert len(ranked) == 3
        assert all(entry["reload"] == mode for entry in ranked)
        records = ledgers[mode]
        assert len(records) == 3
        assert all(record.get("status") is None for record in records)
        assert all(
            record["metrics"].get("routed_accuracy") is not None for record in records
        )

    # V2-1: the hot session never falls back and switches in <= 5 s (median).
    hot_switches = [
        entry["timings"]["switch_s"]
        for entry in run_outputs["hot"]["results"]
        if entry["status"] == "ranked"
    ]
    assert all(switch_s > 0.0 for switch_s in hot_switches)
    assert statistics.median(hot_switches) <= 5.0

    # Identical candidate set -> identical replay outcomes across modes.
    for restart_record, hot_record in zip(
        ledgers["restart"], ledgers["hot"], strict=True
    ):
        assert restart_record["candidate"] == hot_record["candidate"]
        assert restart_record["verdict"] == hot_record["verdict"] == "RANKED"
        assert restart_record["metrics"] == hot_record["metrics"]
        assert restart_record["reasons"] == hot_record["reasons"]
        assert _notes_statuses(restart_record["notes"]) == _notes_statuses(
            hot_record["notes"]
        )
        assert _notes_reload(restart_record["notes"]) == "restart"
        assert _notes_reload(hot_record["notes"]) == "hot"
        assert "fallback_reason" not in hot_record["notes"]
