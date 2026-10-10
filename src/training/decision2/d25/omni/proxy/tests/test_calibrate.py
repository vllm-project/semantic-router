"""Calibration, leave-one-out and margin tests on synthetic boards with known ground truth."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from d25.omni.proxy import board as vb
from d25.omni.proxy import calibrate as cb


def synthetic(
    n: int = 12,
    seed: int = 0,
    proxy_noise: float = 1.0,
    private_noise: float = 1.0,
    useless: tuple[str, ...] = (),
    distorted: tuple[str, ...] = ("CharXiv", "Mind2Web"),
):
    """Board + measurements where each skill is linear in a latent ability plus noise."""
    rng = np.random.default_rng(seed)
    ability = rng.uniform(30, 80, n)
    entrants, refs = [], {}
    for i in range(n):
        bench, public_local, proxy = {}, {}, {}
        for j, b in enumerate(vb.PUBLIC):
            pub = 0.8 * ability[i] + 10 + 3 * j % 7 + rng.normal(0, 1.0)
            local = pub + rng.normal(0, 0.3)
            if b in distorted:
                local = (pub - 5.0) / 1.2 + rng.normal(0, 0.3)
            public_local[b] = float(local)
            entry = {"pub": round(float(pub), 2), "private": None}
            if b in vb.PRIVATE:
                private = 1.1 * ability[i] - 20 + 2 * j + rng.normal(0, private_noise)
                entry["private"] = round(float(private), 2)
                if b in useless:
                    proxy[b] = float(rng.uniform(0, 100))
                else:
                    proxy[b] = float(0.7 * private + 15 + rng.normal(0, proxy_noise))
            bench[b] = entry
        row = {"engine": f"m{i:02d}", "name": f"Model {i}", "bench": bench}
        row.update(vb.recompute(row))
        entrants.append(row)
        refs[row["engine"]] = {"public": public_local, "proxy": proxy}
    board = {
        "generated_utc": "2026-10-09T00:00:00Z",
        "edition": vb.EDITION,
        "weights": {"public": 0.5, "private": 0.5},
        "bench_weights": dict(vb.WEIGHTS),
        "private_sets": list(vb.PRIVATE),
        "entrants": entrants,
        "refs": [],
    }
    return board, {"version": 1, "refs": refs}


def test_t90_table_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 5, 9, 11, 35, 75, 500):
        assert cb.t90(df) == pytest.approx(stats.t.ppf(0.9, df), abs=2e-3)


def test_linear_fit_is_exact_on_noiseless_data():
    x = np.arange(10, dtype=float)[:, None]
    y = 2.0 + 3.0 * x[:, 0]
    model = cb.Linear(("x",)).fit(x, y)
    assert model.intercept == pytest.approx(2.0)
    assert model.coef[0] == pytest.approx(3.0)
    assert cb.loo_rmse(x, y, cb.Linear(("x",))) == pytest.approx(0.0, abs=1e-9)


def test_margin_formula():
    errors = [0.4, -1.0, 0.2, 1.5, -0.3, 0.9, 0.1, -0.6]
    out = cb.margin_from_errors(errors)
    d = np.array(errors)
    n = len(d)
    expected_t = d.mean() + cb.t90(n - 1) * d.std(ddof=1) * math.sqrt(1 + 1 / n)
    assert out["margin_t"] == pytest.approx(expected_t)
    assert out["margin_q90"] == pytest.approx(np.quantile(d, 0.9))
    assert out["margin"] == pytest.approx(max(expected_t, np.quantile(d, 0.9), 0.0))
    assert cb.margin_from_errors([-5.0, -4.0, -6.0])["margin"] == 0.0


def test_informative_proxies_are_selected_and_predict_private():
    board, measurements = synthetic(n=14, proxy_noise=0.5, private_noise=0.5)
    result = cb.calibrate(board, measurements)
    kinds = {b: m["kind"] for b, m in result["private_maps"].items()}
    assert sum(k.startswith("proxy") for k in kinds.values()) >= 7
    assert result["summary"]["rmse"] < 1.0
    assert result["summary"]["spearman_full"] > 0.95


def test_useless_proxy_falls_back_to_public_baseline():
    board, measurements = synthetic(n=14, useless=("Moderation (Hateful Memes)",))
    result = cb.calibrate(board, measurements)
    assert result["private_maps"]["Moderation (Hateful Memes)"]["kind"] == "public"


def test_distorted_public_benchmarks_get_linear_correction_and_exact_ones_do_not():
    board, measurements = synthetic(n=14)
    result = cb.calibrate(board, measurements, exact_public=("CV-Bench",))
    assert result["public_maps"]["CharXiv"]["kind"] == "linear"
    assert result["public_maps"]["CV-Bench"]["kind"] == "identity"
    corr = result["public_maps"]["CharXiv"]["map"]
    assert corr["coef"]["CharXiv"] == pytest.approx(1.2, abs=0.1)


def test_leave_one_out_matches_brute_force():
    board, measurements = synthetic(n=10, seed=3)
    refs = cb.load_refs(board, measurements)
    loo = cb.leave_one_out(refs)
    target = refs[4]
    cal = cb.fit([r for r in refs if r.id != target.id])
    est = cal.estimate(target.public, target.proxy)
    record = next(r for r in loo if r["id"] == target.id)
    assert record["V_hat"] == pytest.approx(est["V_hat"])
    assert record["d"] == pytest.approx(est["V_hat"] - target.full)


def test_aggregate_refs_never_enter_the_fit():
    board, measurements = synthetic(n=10, seed=5)
    base = cb.load_refs(board, measurements)
    extra = dict(measurements)
    first = measurements["refs"]["m00"]
    extra["aggregate_refs"] = {
        "stock": {
            "board": {"pub": 60.0, "priv": 50.0},
            "public": first["public"],
            "proxy": first["proxy"],
        }
    }
    refs = cb.load_refs(board, extra)
    assert cb.fit(base).to_json() == cb.fit(refs).to_json()
    loo = cb.leave_one_out(refs)
    stock = next(r for r in loo if r["id"] == "stock")
    expected = cb.fit(base).estimate(first["public"], first["proxy"])["V_hat"]
    assert stock["V_hat"] == pytest.approx(expected)
    assert stock["d"] == pytest.approx(expected - 55.0)


def test_negative_private_estimates_are_clipped_in_q_hat():
    board, measurements = synthetic(n=10, seed=7)
    cal = cb.fit(cb.load_refs(board, measurements))
    ref = measurements["refs"]["m01"]
    low_proxy = {b: -500.0 for b in vb.PRIVATE}
    est = cal.estimate(ref["public"], low_proxy)
    assert est["Q_hat"] == pytest.approx(vb.private_score(est["private"]))
    assert all(est["private"][b] >= -1e9 for b in vb.PRIVATE)
    assert est["Q_hat"] >= 0.0


def test_missing_proxy_for_some_refs_keeps_the_proxy_model_when_enough_remain():
    board, measurements = synthetic(n=12, proxy_noise=0.3)
    del measurements["refs"]["m00"]["proxy"]["CharXiv"]
    result = cb.calibrate(board, measurements)
    assert result["private_maps"]["CharXiv"]["kind"].startswith("proxy")


def test_calibration_json_roundtrip_gives_identical_estimates():
    board, measurements = synthetic(n=11, seed=9)
    cal = cb.fit(cb.load_refs(board, measurements))
    again = cb.Calibration.from_json(json.loads(json.dumps(cal.to_json())))
    ref = measurements["refs"]["m03"]
    assert again.estimate(ref["public"], ref["proxy"])["V_hat"] == pytest.approx(
        cal.estimate(ref["public"], ref["proxy"])["V_hat"]
    )


def test_spearman_handles_ties():
    assert cb.spearman([1, 2, 2, 3], [10, 20, 20, 30]) == pytest.approx(1.0)
    assert cb.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


@pytest.mark.skipif(
    not os.environ.get("D25_VISION_JSON"), reason="set D25_VISION_JSON to a board file"
)
def test_published_board_follows_the_aggregation_rule():
    board = vb.load(Path(os.environ["D25_VISION_JSON"]))
    assert vb.audit(board) == []
