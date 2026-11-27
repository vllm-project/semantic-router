"""Splitting tests (B): deterministic group-level 3-way split."""

from __future__ import annotations

import json

import pytest
from recipe_lab.contract import ContractError
from recipe_lab.splitting import (
    default_split_rule,
    group_cases,
    load_cases,
    parse_ratios,
    split_cases,
    write_splits,
)


def _cases(n: int, groups: int = 4) -> list[dict]:
    return [
        {
            "case_id": f"c{i}",
            "prompt": f"prompt {i}",
            "truth": "math",
            "source_id": f"src{i % groups}",
        }
        for i in range(n)
    ]


def test_same_seed_same_output(tmp_path):
    # determinism contract: identical inputs => byte-identical slice files
    digests = []
    for run_dir in (tmp_path / "a", tmp_path / "b"):
        cases = _cases(40)
        split_map = split_cases(cases, (0.4, 0.3, 0.3), "source_id", 42)
        splits = write_splits(split_map, run_dir)
        digests.append({name: spec["content_hash"] for name, spec in splits.items()})
    assert digests[0] == digests[1]


def test_groups_never_span_two_splits():
    cases = _cases(40, groups=8)
    split_map = split_cases(cases, (0.4, 0.3, 0.3), "source_id", 42)
    seen: dict[str, str] = {}
    for name, rows in split_map.items():
        for row in rows:
            assert (
                seen.setdefault(row["source_id"], name) == name
            ), "source group leaked across splits"
    assert set(split_map) == {"calib", "dev", "frozen_test"}


def test_missing_group_key_falls_back_to_singleton_groups():
    cases = [{"case_id": f"c{i}", "prompt": f"p{i}", "truth": "math"} for i in range(6)]
    split_map = split_cases(cases, (0.5, 0.25, 0.25), "source_id", 42)
    total = sum(len(rows) for rows in split_map.values())
    assert total == 6  # every case lands somewhere


def test_parse_ratios_rejects_bad_input():
    with pytest.raises(ContractError, match="three"):
        parse_ratios("0.5,0.5")
    with pytest.raises(ContractError, match=r"sum to 1.0"):
        parse_ratios("0.5,0.5,0.5")
    assert parse_ratios("0.4,0.3,0.3") == (0.4, 0.3, 0.3)


def test_load_cases_requires_case_id_and_prompt(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"case_id": "c1", "prompt": "p"})
        + "\n"
        + json.dumps({"case_id": "c2"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="prompt"):
        load_cases(path)


def test_default_split_rule_text():
    assert (
        default_split_rule("source_id", 42, (0.4, 0.3, 0.3))
        == "group by source_id; seed 42; 40/30/30"
    )


def test_group_cases_orders_deterministically():
    cases = _cases(6, groups=3)
    groups = group_cases(cases, "source_id")
    assert sorted(groups) == ["src0", "src1", "src2"]
    assert [c["case_id"] for c in groups["src0"]] == ["c0", "c3"]
