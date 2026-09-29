import json
import math

import pytest

from v2.eval.htdev import validate as V


def model(key, tier, h_formal, h_dev, h_pilot, h_mean3=None, v3=None, t_dev=0.5):
    h_mean3 = h_pilot if h_mean3 is None else h_mean3
    v3 = 100 * h_formal if v3 is None else v3
    return {
        "key": key,
        "tier": tier,
        "group": "peer",
        "formal": {"H_formal": h_formal, "css15_task_mean": h_formal + 0.01, "v3": v3},
        "htdev": {"H_dev": h_dev, "H_dev_mean": h_dev, "H_dev_sd": 0.01},
        "development": {
            "T_dev": t_dev,
            "H_pilot": h_pilot,
            "H_mean3": h_mean3,
            "noise_sd": {"T_dev": 0.01, "H_pilot": 0.02, "H_mean3": 0.015, "P": 1.0},
        },
        "proxies": {
            "P": 100 * math.sqrt(t_dev * h_pilot),
            "P_HT": 100 * math.sqrt(t_dev * h_dev),
            "P_HT_mean": 100 * math.sqrt(t_dev * h_dev),
        },
    }


def rows_of(models):
    return [V.flatten(m) for m in models]


def test_pair_score_ties_count_half():
    assert V.pair_score(0.0, 0.1) == 0.5
    assert V.pair_score(0.2, 0.1) == 1.0
    assert V.pair_score(-0.2, 0.1) == 0.0


def test_agreement_threshold_and_within_tier_only():
    ms = [
        model("a", "4B", 0.50, 0.40, 0.30),
        model("b", "4B", 0.53, 0.45, 0.20),  # dev agrees, pilot disagrees (dH=0.03)
        model("c", "4B", 0.51, 0.45, 0.40),  # vs a dH=0.01 (below 0.02), vs b dH=-0.02
        model("z", "9B", 0.90, 0.10, 0.90),  # other tier: never paired
    ]
    rows = rows_of(ms)
    pairs = V.within_pairs(rows)
    assert len(pairs) == 3
    dev = V.agreement(rows, pairs, "H_dev", "H_formal", 0.02)
    # pairs with |dH| >= 0.02: (a,b) dev +; (b,c) dH=0.02, dev tie -> 0.5
    assert dev["n"] == 2 and dev["score"] == pytest.approx(1.5)
    pilot = V.agreement(rows, pairs, "H_pilot", "H_formal", 0.02)
    # (a,b): pilot drops -> 0; (b,c): pilot rises while H falls -> 0
    assert pilot["score"] == 0.0


def test_duplicate_draws_form_no_pair():
    ms = [model("a", "2B", 0.4, 0.4, 0.4), model("b", "2B", 0.5, 0.5, 0.5)]
    rows = rows_of([ms[0], ms[0], ms[1]])
    assert V.within_pairs(rows) == [(0, 2), (1, 2)]


def test_bootstrap_is_seeded_and_perfect_dev_wins():
    ms = []
    for tier, base in (("0.6B", 0.3), ("4B", 0.5)):
        for k in range(5):
            h = base + 0.03 * k
            pilot = base + 0.03 * ((k * 3) % 5)  # scrambled order
            ms.append(model(f"{tier}-{k}", tier, h, h - 0.1, pilot))
    rows = rows_of(ms)
    one = V.paired_bootstrap(rows, 200, 7)
    two = V.paired_bootstrap(rows, 200, 7)
    assert one == two
    assert one["H_pilot"]["primary"]["p_better"] > 0.9


def test_analyze_decision_and_tie_bands():
    ms = []
    for tier, base in (("0.6B", 0.30), ("0.8B", 0.40), ("4B", 0.50)):
        for k in range(4):
            h = base + 0.025 * k
            wobble = 0.004 * ((k * 7) % 3 - 1)
            ms.append(
                model(
                    f"{tier}-{k}",
                    tier,
                    h,
                    h - 0.05 + wobble,
                    base + 0.01 * (3 - k),
                    v3=30 + 60 * h + 3 * wobble,
                )
            )
    result = V.analyze({"models": ms}, draws=300)
    decision = result["section5"]["decision"]
    assert decision["H_dev_primary"] == 1.0
    assert decision["tracks_better"] is True
    band = result["section5"]["h_dev_tie_band"]
    assert band["band"] is not None and band["band"] >= 0.005
    assert round(band["band"] / 0.005, 6) == int(round(band["band"] / 0.005))
    tie = result["section6"]["proxies"]["P_HT"]["tie_band"]
    assert tie["band"] is None or tie["band"] >= 1
    assert result["section6"]["status"].startswith("binding")
    assert len(result["table"]) == 12
    json.dumps(result, allow_nan=False)
    assert result["section5"]["agreement_by_H_dev_gap"][-1]["bin"][1] is None
