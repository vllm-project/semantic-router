from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit


def test_official_baseline():
    fx = audit.load("official_authz_premium_complex.json")
    r = audit.baseline_reproduction(fx)
    assert r["status"] == "PASS"
    assert r["actual_decision"] == "premium_complex"


def test_official_counterfactual():
    fx = audit.load("official_authz_premium_complex.json")
    r = audit.discrete_counterfactual_margin(fx)
    assert r["status"] == "FLIP"
    assert r["margin"] == 1
    assert r["counterfactual_decision"] == "premium_default"


def test_gray_margin():
    fx = audit.load("synthetic_gray_boundary.json")
    r = audit.numeric_boundary_margin(fx)
    assert r["status"] == "CRITICAL"
    assert abs(r["margin"] - 0.05) < 1e-9


def test_order_flip():
    fx = audit.load("synthetic_order.json")
    r = audit.order_commutator(fx)
    assert r["status"] == "FAIL"
    assert r["A_then_B_route"] == "normal_route"
    assert r["B_then_A_route"] == "guarded_route"


def test_grouping_flip():
    fx = audit.load("synthetic_grouping.json")
    r = audit.grouping_associator(fx)
    assert r["status"] == "FAIL"
    assert r["left"]["route"] == "normal_route"
    assert r["right"]["route"] == "guarded_route"
    assert abs(r["associator_residual"] - 0.05) < 1e-9
