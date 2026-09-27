"""Small contract tests; the production candidate uses a separate private seed."""

from __future__ import annotations

import collections

import pytest

from training.data import score_v84_audit as audit
from training.data import score_v84_pilot as pilot


def test_cardinality_language_and_oracles() -> None:
    seed = bytes(range(32))
    train = pilot.build(seed, "train")
    select = pilot.build(seed, "select")
    assert (len(train), len(select)) == (54, 36)
    assert collections.Counter(row["language"] for row in train + select) == {
        "en": 54,
        "zh": 36,
    }
    groups = collections.defaultdict(list)
    for row in train + select:
        groups[row["group_id"]].append(row)
        meta = row["audit_metadata"]
        assert (
            pilot.rendered_oracle(row["state"], meta["mechanism"], meta["case"])
            == row["label"]
        )
    assert len(groups) == 30
    assert all(
        {r["label"] for r in triplet} == {0, 1, 2} for triplet in groups.values()
    )
    assert set(r["group_id"] for r in train).isdisjoint(r["group_id"] for r in select)
    assert {r["audit_metadata"]["mechanism"] for r in train} == set(pilot.MECHANISMS)


def test_cutoff_equality_is_feasible() -> None:
    rng = pilot._rng(bytes(range(32)), "train", "connection", 0)
    base = pilot._base(rng, "train", "connection", 0)
    facts = pilot._facts(base, 2)
    assert facts["latest"] + facts["walk"] == facts["cutoff"]
    assert pilot.oracle(facts) == 2
    assert pilot.rendered_oracle(pilot.render(facts), "connection", base["case"]) == 2


def test_workflow_never_completes_descendant_of_failed_or_queued_ancestor() -> None:
    rng = pilot._rng(bytes(range(32)), "train", "workflow", 0)
    base = pilot._base(rng, "train", "workflow", 0)
    for level in pilot.LEVELS:
        facts = pilot._facts(base, level)
        if facts["a_status"] != "complete":
            assert facts["b_status"] != "complete"
        assert pilot.oracle(facts) == level


def test_missing_required_rendered_evidence_rejected() -> None:
    rng = pilot._rng(bytes(range(32)), "train", "evidence", 0)
    base = pilot._base(rng, "train", "evidence", 0)
    state = pilot.render(pilot._facts(base, 2))
    state = state.replace("Stability [", "Removed [")
    with pytest.raises(ValueError, match="Missing rendered evidence"):
        pilot.rendered_oracle(state, "evidence", base["case"])


def test_text_normalization_preserves_chinese_and_masks_ids() -> None:
    first = audit._clean("案号 CASE-ABCD123，目标 ITEM-ZYXW456，12 件")
    second = audit._clean("案号 CASE-QWER987，目标 ITEM-ASDF654，23 件")
    assert first == second
    assert audit._ngrams(first)
