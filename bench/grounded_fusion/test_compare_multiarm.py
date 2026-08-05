"""Unit tests for the multi-arm verdict logic (no model/router needed).

Run: .venv-bench/bin/python -m pytest bench/grounded_fusion/test_compare_multiarm.py -q
"""

import json

from bench.grounded_fusion.compare_multiarm import compare


def _write_arm(d, arm, finals, errors=None):
    """Write a synthetic samples_{arm}.jsonl. finals: {id: normalized_score}."""
    errors = errors or {}
    path = d / f"samples_{arm}.jsonl"
    with path.open("w") as fh:
        for sid, norm in finals.items():
            rec = {
                "id": sid,
                "domain": "Medicine",
                "error": errors.get(sid),
                "final": {
                    "total": norm * 100,
                    "normalized": norm,
                    "negative_penalty": 0.0,
                    "n_negative_triggered": 0,
                    "per_section": {},
                },
                "panel": [],
            }
            fh.write(json.dumps(rec) + "\n")


def _ids(n):
    return [f"q{i}" for i in range(n)]


def test_keep_grounding(tmp_path):
    # C beats B and D by a constant margin; B beats A. Constant deltas -> tight CI.
    ids = _ids(12)
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.4))
    _write_arm(tmp_path, "B", dict.fromkeys(ids, 0.55))
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.7))
    _write_arm(tmp_path, "D", dict.fromkeys(ids, 0.58))
    report = compare(str(tmp_path), ["A", "B", "C", "D"])
    assert report["decision"] == "KEEP_GROUNDING"
    assert report["pairs"]["C_vs_B"]["normalized"]["favorable"]
    assert report["pairs"]["C_vs_D"]["normalized"]["favorable"]


def test_kill_grounding_addon_when_c_equals_b(tmp_path):
    # C ~= B (zero delta -> CI is [0,0], not favorable).
    ids = _ids(12)
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.4))
    _write_arm(tmp_path, "B", dict.fromkeys(ids, 0.6))
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.6))
    _write_arm(tmp_path, "D", dict.fromkeys(ids, 0.55))
    report = compare(str(tmp_path), ["A", "B", "C", "D"])
    assert report["decision"] == "KILL_GROUNDING_ADDON"


def test_kill_grounding_addon_when_c_equals_placebo(tmp_path):
    # C beats B, but does NOT beat the random-weight placebo D.
    ids = _ids(12)
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.4))
    _write_arm(tmp_path, "B", dict.fromkeys(ids, 0.55))
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.7))
    _write_arm(tmp_path, "D", dict.fromkeys(ids, 0.7))
    report = compare(str(tmp_path), ["A", "B", "C", "D"])
    assert report["decision"] == "KILL_GROUNDING_ADDON"
    assert "placebo" in report["rationale"]


def test_kill_fusion_when_solo_beats_fusion(tmp_path):
    # A (one model) significantly beats B (plain fusion).
    ids = _ids(12)
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.7))
    _write_arm(tmp_path, "B", dict.fromkeys(ids, 0.5))
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.55))
    _write_arm(tmp_path, "D", dict.fromkeys(ids, 0.52))
    report = compare(str(tmp_path), ["A", "B", "C", "D"])
    assert report["decision"] == "KILL_FUSION"


def test_inconclusive_when_fusion_vs_solo_unproven(tmp_path):
    # B-A straddles 0 (mixed signs) -> fusion vs solo unproven.
    ids = _ids(12)
    mixed = {i: (0.6 if int(i[1:]) % 2 == 0 else 0.4) for i in ids}
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.5))
    _write_arm(tmp_path, "B", mixed)
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.7))
    _write_arm(tmp_path, "D", dict.fromkeys(ids, 0.55))
    report = compare(str(tmp_path), ["A", "B", "C", "D"])
    assert report["decision"] == "INCONCLUSIVE"


def test_strict_paired_set_excludes_errored_ids(tmp_path):
    ids = _ids(12)
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.4))
    _write_arm(tmp_path, "B", dict.fromkeys(ids, 0.55))
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.7), errors={"q0": "HTTP 500"})
    _write_arm(tmp_path, "D", dict.fromkeys(ids, 0.58))
    report = compare(str(tmp_path), ["A", "B", "C", "D"])
    # q0 errored in arm C -> dropped from the paired set.
    assert report["n_paired"] == 11


def test_missing_placebo_is_inconclusive(tmp_path):
    ids = _ids(12)
    _write_arm(tmp_path, "A", dict.fromkeys(ids, 0.4))
    _write_arm(tmp_path, "B", dict.fromkeys(ids, 0.55))
    _write_arm(tmp_path, "C", dict.fromkeys(ids, 0.7))
    report = compare(str(tmp_path), ["A", "B", "C"])
    assert report["decision"] == "INCONCLUSIVE"
    assert "placebo" in report["rationale"]
