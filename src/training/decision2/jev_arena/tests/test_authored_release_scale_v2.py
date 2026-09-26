"""Prospective v2 distribution and snapshot gates."""

from __future__ import annotations

import pytest

from jev_arena.authored_release_scale_v2 import _distribution


def test_v2_balanced_native_targets_include_joint_hold() -> None:
    cases = (
        [
            {
                "slug": f"c{i}",
                "option_order": ["A", "B", "C", "HOLD"],
            }
            for i in range(4)
        ]
        + [{"slug": f"n{i}"} for i in range(4)]
        + [{"slug": f"s{i}"} for i in range(4)]
    )
    proofs = (
        [{"type": "choice", "original": answer} for answer in ("A", "B", "C", "HOLD")]
        + [
            {"type": "noul", "original": answer}
            for answer in (True, False, True, False)
        ]
        + [{"type": "score", "original": answer} for answer in (0, 1, 2, 0)]
    )
    result = _distribution(cases, proofs)
    assert result["choice_hold_answers"] == 1
    assert result["noul_false"] == 2
    assert result["score_levels"] == {"0": 2, "1": 1, "2": 1}
    proofs[3]["original"] = "A"
    with pytest.raises(ValueError, match="Choice"):
        _distribution(cases, proofs)
