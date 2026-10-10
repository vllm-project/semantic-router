"""Gate decisions against a mocked live board."""

from __future__ import annotations

import pytest

from d25.omni.proxy import board as vb
from d25.omni.proxy import calibrate as cb
from d25.omni.proxy import gate
from d25.omni.proxy.tests.test_calibrate import synthetic


def live_board(fulls: list[float], pubs: list[float]) -> dict:
    entrants = []
    for i, (full, pub) in enumerate(zip(fulls, pubs)):
        priv = 2 * full - pub
        bench = {
            b: {"pub": pub, "private": priv if b in vb.PRIVATE else None}
            for b in vb.PUBLIC
        }
        entrants.append(
            {
                "engine": f"e{i}",
                "name": f"E{i}",
                "bench": bench,
                "pub": pub,
                "priv": priv,
                "full": full,
            }
        )
    return {
        "generated_utc": "2026-10-09T12:43:49Z",
        "edition": vb.EDITION,
        "weights": {"public": 0.5, "private": 0.5},
        "bench_weights": dict(vb.WEIGHTS),
        "private_sets": list(vb.PRIVATE),
        "entrants": entrants,
        "refs": [],
    }


def fixed_calibration(margin: float) -> dict:
    """Identity public maps and private maps that copy the proxy, so V_hat is known exactly."""
    board, measurements = synthetic(n=8)
    cal = cb.fit(cb.load_refs(board, measurements)).to_json()
    for b in vb.PUBLIC:
        cal["public_maps"][b] = {
            "kind": "identity",
            "map": cb.Linear((b,), 0.0, (1.0,), 0).to_json(),
            "loo_rmse": {},
        }
    for b in vb.PRIVATE:
        name = f"proxy:{b}"
        cal["private_maps"][b] = {
            "kind": "proxy",
            "map": cb.Linear((name,), 0.0, (1.0,), 8).to_json(),
            "loo_rmse": {},
            "ranges": {name: [0.0, 100.0]},
            "fallback": cb.Linear((f"public:{b}",), 0.0, (1.0,), 8).to_json(),
        }
    cal["summary"] = {"margin": margin}
    cal["refs"] = ["m00"]
    return cal


def candidate(public: float, private: float) -> dict:
    return {
        "name": "cand",
        "public": {b: public for b in vb.PUBLIC},
        "proxy": {b: private for b in vb.PRIVATE},
    }


def test_private_push_but_not_public_submit():
    live = live_board([70.58, 69.60, 66.83, 64.18], [73.21, 72.80, 69.88, 70.21])
    result = gate.decide(fixed_calibration(margin=2.0), candidate(72.0, 66.0), live)
    assert result["V_hat"] == pytest.approx(69.0)
    assert result["private_push"] and result["private_push_strict"]
    assert not result["public_submit"]
    assert result["rank_at_lower_bound"] == 3
    assert result["need_for_private_push"] == pytest.approx(68.83)


def test_strict_variant_requires_public_above_third():
    live = live_board([70.58, 69.60, 66.83, 64.18], [73.21, 72.80, 69.88, 70.21])
    result = gate.decide(fixed_calibration(margin=1.0), candidate(69.0, 70.0), live)
    assert result["private_push"]
    assert not result["private_push_strict"]


def test_public_submit_when_lower_bound_beats_first():
    live = live_board([70.58, 69.60, 66.83], [73.21, 72.80, 69.88])
    result = gate.decide(fixed_calibration(margin=1.5), candidate(75.0, 70.0), live)
    assert result["public_submit"] and result["public_submit_strict"]
    assert result["rank_at_lower_bound"] == 1
    assert "public + submit: YES" in gate.report(result)


def test_missing_proxy_uses_the_public_fallback_and_is_reported():
    live = live_board([70.58, 69.60, 66.83], [73.21, 72.80, 69.88])
    cand = candidate(70.0, 60.0)
    del cand["proxy"]["Winoground"]
    result = gate.decide(fixed_calibration(margin=1.0), cand, live)
    assert result["fallbacks"] == ["Winoground"]
    assert result["private_estimates"]["Winoground"] == pytest.approx(70.0)


def test_extrapolation_is_flagged():
    live = live_board([70.58, 69.60, 66.83], [73.21, 72.80, 69.88])
    result = gate.decide(fixed_calibration(margin=1.0), candidate(70.0, 140.0), live)
    assert result["extrapolation"]


def test_board_layout_change_is_rejected():
    live = live_board([70.0, 69.0, 68.0], [70.0, 69.0, 68.0])
    live["bench_weights"]["MMMU-Pro vision"] = 0.5
    with pytest.raises(ValueError):
        vb.check_layout(live)
