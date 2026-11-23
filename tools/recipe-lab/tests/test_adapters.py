"""Router Replay adapter tests: fetch mocking + prompt-join semantics."""

from __future__ import annotations

import json
import urllib.error
from unittest import mock

import pytest
from recipe_lab.adapters.router_replay import (
    fetch_replays,
    load_cases,
    record_to_case,
    records_to_cases,
    records_to_rows,
)
from recipe_lab.contract import ContractError


def fake_record(request_id: str, prompt: str, decision: str, selected: str) -> dict:
    return {
        "request_id": request_id,
        "decision": decision,
        "selected_model": selected,
        "prompt": prompt,
        "signals": {"decision": ["depth"]},
    }


def evidence_record(request_id: str, prompt: str, decision: str, selected: str) -> dict:
    """A record shaped like the real router wire format (camelCase keys)."""
    return {
        "RequestID": request_id,
        "Decision": decision,
        "SelectedModel": selected,
        "Prompt": prompt,
        "SignalValues": {
            "decision:difficulty": 1.75,
            "decision:domain:code": 0.97,
            "decision:domain:math": 0.01,
        },
        "SignalConfidences": {"decision:domain": 0.97},
        "DecisionTier": 2,
    }


CASES = [
    {
        "case_id": "c1",
        "prompt": "What is the capital of France?",
        "truth": "local_standard",
    },
    {
        "case_id": "c2",
        "prompt": "Analyse quicksort vs mergesort from first principles.",
        "truth": "cloud_frontier_reasoning",
    },
]


def test_record_to_case_joins_by_prompt():
    row, state = record_to_case(
        fake_record(
            "r1",
            "What is the capital of France?",
            "local_standard",
            "local/private-qwen",
        ),
        {c["prompt"]: type("C", (), c) for c in CASES},
    )
    assert state == "matched"
    assert row["case_id"] == "c1"
    assert row["truth"] == "local_standard"
    assert row["matched_decision"] == "local_standard"  # F1: pred -> matched_decision
    assert row["routed"] == "local/private-qwen"
    assert row["answer_ok"] is None  # correctness joins later


def test_record_to_case_skip_reasons():
    by_prompt = {c["prompt"]: type("C", (), c) for c in CASES}
    row, state = record_to_case(fake_record("r1", "", "d", "m"), by_prompt)
    assert row is None and state == "no_prompt"
    row, state = record_to_case(
        fake_record("r2", "unlabeled prompt", "d", "m"), by_prompt
    )
    assert row is None and state == "unlabeled"


def test_records_to_cases_dedup_and_stats():
    records = [
        fake_record(
            "r1",
            "What is the capital of France?",
            "local_standard",
            "local/private-qwen",
        ),
        fake_record(
            "r2",
            "What is the capital of France?",
            "local_standard",
            "local/private-qwen",
        ),
        fake_record(
            "r3",
            "Analyse quicksort vs mergesort from first principles.",
            "local_standard",
            "local/private-qwen",
        ),
        {"request_id": "r4"},
    ]
    rows, stats = records_to_cases(records, [type("C", (), c) for c in CASES])
    assert len(rows) == 2
    assert rows[0]["case_id"] == "c1"
    assert rows[1]["case_id"] == "c2"
    assert stats == {
        "matched": 2,
        "no_prompt": 1,
        "unlabeled": 0,
        "duplicate_prompt": 1,
    }


def test_record_to_case_passes_signal_evidence():
    by_prompt = {c["prompt"]: type("C", (), c) for c in CASES}
    row, state = record_to_case(
        evidence_record(
            "r1",
            "What is the capital of France?",
            "local_standard",
            "local/private-qwen",
        ),
        by_prompt,
    )
    assert state == "matched"
    # signal evidence is carried verbatim for later diagnostics (D adapter)
    assert row["signal_values"]["decision:difficulty"] == 1.75
    assert (
        row["signal_values"]["decision:domain:code"] == 0.97
    )  # choice-type: per-choice key
    assert row["signal_confidences"] == {"decision:domain": 0.97}
    assert row["decision_tier"] == 2
    assert row["matched_decision"] == "local_standard"


def test_records_to_rows_flattens_every_record():
    # unlabeled path: no join, no skip; records without a prompt are kept
    records = [
        evidence_record("r1", "prompt one", "local_standard", "m1"),
        {"RequestID": "r2", "Decision": "cloud_frontier"},  # no prompt at all
    ]
    rows = records_to_rows(records)
    assert len(rows) == 2
    assert rows[0]["matched_decision"] == "local_standard"
    assert rows[0]["signal_values"]["decision:domain:math"] == 0.01
    assert rows[1]["request_id"] == "r2"
    assert rows[1]["prompt"] == ""
    assert rows[1]["signal_values"] == {}
    assert rows[1]["decision_tier"] is None


def test_fetch_replays_follows_pagination():
    # 3 pages: the fetcher must follow next_offset until has_more is false
    pages = [
        {
            "data": [
                fake_record("r1", "p1", "d", "m"),
                fake_record("r2", "p2", "d", "m"),
            ],
            "count": 2,
            "total": 3,
            "has_more": True,
            "next_offset": 2,
        },
        {
            "data": [fake_record("r3", "p3", "d", "m")],
            "count": 1,
            "total": 3,
            "has_more": True,
            "next_offset": 3,
        },
        {"data": [], "count": 0, "total": 3, "has_more": False},
    ]
    response = mock.MagicMock()
    response.__enter__ = lambda s: s
    response.__exit__ = lambda s, *a: None
    response.read.side_effect = [json.dumps(p).encode() for p in pages]
    with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
        records = fetch_replays("http://127.0.0.1:8080")
    assert [r["request_id"] for r in records] == ["r1", "r2", "r3"]
    requested = [call.args[0].full_url for call in urlopen.call_args_list]
    assert "limit=50&offset=0" in requested[0] and "showDetails=true" in requested[0]
    assert "offset=2" in requested[1]
    assert "offset=3" in requested[2]


def test_fetch_replays_halves_page_on_http_413():
    # A page of detailed records can overflow the management API's
    # response-body cap (HTTP 413); the fetcher must retry the same offset
    # with a halved page instead of failing the whole fetch.
    page = {"data": [fake_record("r1", "p1", "d", "m")], "total": 1, "has_more": False}
    response = mock.MagicMock()
    response.__enter__ = lambda s: s
    response.__exit__ = lambda s, *a: None
    response.read.return_value = json.dumps(page).encode()

    def urlopen_side_effect(request, *args, **kwargs):
        if "limit=50" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 413, "too large", {}, None)
        return response

    with mock.patch(
        "urllib.request.urlopen", side_effect=urlopen_side_effect
    ) as urlopen:
        records = fetch_replays("http://127.0.0.1:8080")
    assert [r["request_id"] for r in records] == ["r1"]
    requested = [call.args[0].full_url for call in urlopen.call_args_list]
    assert "limit=50&offset=0" in requested[0]
    assert "limit=25&offset=0" in requested[1]


def test_load_cases_rejects_missing_truth(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"case_id": "c1", "prompt": "p"}), encoding="utf-8")
    with pytest.raises(ContractError, match="truth"):
        load_cases(path)


def test_collect_replays_unlabeled_cli(tmp_path, monkeypatch):
    # --unlabeled skips the join and writes one raw row per record
    from recipe_lab import cli

    out = tmp_path / "rows.jsonl"
    monkeypatch.setattr(
        cli, "fetch_replays", lambda *a, **k: [evidence_record("r1", "p1", "d", "m")]
    )
    rc = cli.main(
        [
            "collect-replays",
            "--url",
            "http://127.0.0.1:8080",
            "--unlabeled",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["request_id"] == "r1"
    assert rows[0]["matched_decision"] == "d"


def test_collect_replays_flag_rules():
    from recipe_lab import cli

    with pytest.raises(SystemExit, match="mutually exclusive"):
        cli.main(
            [
                "collect-replays",
                "--url",
                "http://x",
                "--unlabeled",
                "--cases",
                "c.jsonl",
                "--output",
                "o.jsonl",
            ]
        )
    with pytest.raises(SystemExit, match=r"one of --cases|--unlabeled|is required"):
        cli.main(["collect-replays", "--url", "http://x", "--output", "o.jsonl"])


def test_fetch_replays_parses_list():
    payload = {"data": [fake_record("r1", "p", "d", "m")], "total": 1}
    response = mock.MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__ = lambda s: s
    response.__exit__ = lambda s, *a: None
    with mock.patch("urllib.request.urlopen", return_value=response):
        records = fetch_replays("http://127.0.0.1:8080")
    assert len(records) == 1 and records[0]["request_id"] == "r1"
