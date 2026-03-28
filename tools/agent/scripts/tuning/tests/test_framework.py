"""Tests for the DSL tuning framework core components.

Validates scenario contracts, engine diagnostics, offline analyzer,
probe loading, and CLI scenario discovery — all without a live router.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DSL_CONFIG = {
    "routing": {
        "projections": {
            "scores": [
                {
                    "name": "privacy_risk_score",
                    "method": "weighted_sum",
                    "inputs": [
                        {
                            "name": "pii_detected",
                            "type": "pii",
                            "weight": 0.6,
                        },
                        {
                            "name": "internal_doc",
                            "type": "keyword",
                            "weight": 0.4,
                            "value_source": "confidence",
                        },
                    ],
                }
            ],
            "mappings": [
                {
                    "name": "privacy_policy_band",
                    "source": "privacy_risk_score",
                    "outputs": [
                        {"name": "policy_privacy_cloud", "lt": 0.35},
                        {"name": "policy_privacy_local", "gte": 0.35},
                    ],
                }
            ],
        },
        "decisions": [
            {
                "name": "local_privacy",
                "priority": 200,
                "rules": {
                    "operator": "AND",
                    "conditions": [
                        {"type": "projection", "name": "policy_privacy_local"}
                    ],
                },
                "modelRefs": [{"model": "local/private"}],
            },
            {
                "name": "cloud_standard",
                "priority": 100,
                "rules": {
                    "operator": "AND",
                    "conditions": [
                        {"type": "projection", "name": "policy_privacy_cloud"}
                    ],
                },
                "modelRefs": [{"model": "cloud/standard"}],
            },
        ],
    }
}

SAMPLE_PROBES_FLAT = [
    {
        "id": "p1",
        "query": "Show me PII data",
        "expected_decision": "local_privacy",
        "tags": ["privacy"],
    },
    {
        "id": "p2",
        "query": "Hello world",
        "expected_decision": "cloud_standard",
        "tags": ["baseline"],
    },
]

SAMPLE_PROBES_GROUPED = {
    "decisions": [
        {
            "id": "local_privacy",
            "expected_decision": "local_privacy",
            "tags": ["privacy"],
            "probes": [
                {"id": "p1", "query": "Show me PII data"},
                {"id": "p2", "query": "Internal doc review"},
            ],
        },
        {
            "id": "cloud_standard",
            "expected_decision": "cloud_standard",
            "tags": ["baseline"],
            "probes": [
                {"id": "p3", "query": "What is 2+2?"},
            ],
        },
    ]
}


@pytest.fixture
def dsl_config():
    from tuning.engine import load_dsl_config

    return load_dsl_config(SAMPLE_DSL_CONFIG)


@pytest.fixture
def tmp_probes_flat(tmp_path):
    p = tmp_path / "probes.yaml"
    p.write_text(yaml.dump(SAMPLE_PROBES_FLAT))
    return p


@pytest.fixture
def tmp_probes_grouped(tmp_path):
    p = tmp_path / "probes.yaml"
    p.write_text(yaml.dump(SAMPLE_PROBES_GROUPED))
    return p


# ---------------------------------------------------------------------------
# Scenario ABC contract
# ---------------------------------------------------------------------------


class TestScenarioContract:
    def test_cannot_instantiate_abc(self):
        from tuning.scenario import Scenario

        with pytest.raises(TypeError):
            Scenario()

    def test_must_implement_name(self):
        from tuning.scenario import Scenario

        class Incomplete(Scenario):
            def severity(self, probe):
                return 1

        with pytest.raises(TypeError):
            Incomplete()

    def test_must_implement_severity(self):
        from tuning.scenario import Scenario

        class Incomplete(Scenario):
            @property
            def name(self):
                return "test"

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_scenario(self):
        from tuning.scenario import Scenario

        class Minimal(Scenario):
            @property
            def name(self):
                return "minimal_test"

            def severity(self, probe):
                return 5

        s = Minimal()
        assert s.name == "minimal_test"
        assert s.severity({}) == 5
        assert s.adapt_result({}, {}) is None
        assert s.build_output({"key": "val"}) == {"key": "val"}


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------


class TestBuiltinScenarios:
    def test_privacy_scenario_interface(self):
        from tuning.scenarios.privacy import PrivacyScenario

        s = PrivacyScenario()
        assert isinstance(s.name, str) and len(s.name) > 0
        assert isinstance(s.severity({"tags": ["security"], "expected": "x"}), int)

    def test_calibration_scenario_interface(self):
        from tuning.scenarios.calibration import CalibrationScenario

        s = CalibrationScenario()
        assert isinstance(s.name, str) and len(s.name) > 0
        assert isinstance(s.severity({"tags": ["biology"]}), int)

    def test_calibration_severity_by_uplift(self):
        from tuning.scenarios.calibration import CalibrationScenario

        s = CalibrationScenario()
        assert s.severity({"tags": ["computer_science"]}) == 10  # net=8
        assert s.severity({"tags": ["business"]}) == 5  # net=3
        assert s.severity({"tags": ["chemistry"]}) == 3  # net=1
        assert s.severity({"tags": ["history"]}) == 1  # net=0

    def test_calibration_adapt_result_none_as_keep_7b(self):
        from tuning.scenarios.calibration import CalibrationScenario

        s = CalibrationScenario()
        probe = {
            "id": "q1",
            "query": "test",
            "expected_decision": "keep_7b",
            "tags": [],
        }
        resp = {"decision_result": {"decision_name": "NONE"}, "signal_confidences": {}}
        result = s.adapt_result(probe, resp)
        assert result["correct"] is True
        assert result["actual"] == "keep_7b"

    def test_privacy_severity_weights(self):
        from tuning.scenarios.privacy import PrivacyScenario

        s = PrivacyScenario()
        assert s.severity({"tags": ["security"], "expected": "x"}) == 10
        assert s.severity({"tags": ["privacy"], "expected": "x"}) == 10
        assert s.severity({"tags": ["baseline"], "expected": "x"}) == 1
        assert s.severity({"tags": [], "expected": "x"}) == 3

    def test_confidence_helpers(self):
        from tuning.scenarios.confidence import normalize_logprob

        assert normalize_logprob(0.0) == 1.0
        assert normalize_logprob(-3.0) == 0.0
        assert normalize_logprob(-1.5) == 0.5
        assert normalize_logprob(-10.0) == 0.0
        assert normalize_logprob(5.0) == 1.0


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------


class TestProbeLoading:
    def test_load_flat(self, tmp_probes_flat):
        from tuning.probes import load_probes

        probes = load_probes(tmp_probes_flat)
        assert len(probes) == 2
        assert probes[0]["id"] == "p1"
        assert probes[1]["expected_decision"] == "cloud_standard"

    def test_load_grouped(self, tmp_probes_grouped):
        from tuning.probes import load_probes

        probes = load_probes(tmp_probes_grouped)
        assert len(probes) == 3
        assert all("expected_decision" in p for p in probes)
        assert probes[0]["expected_decision"] == "local_privacy"
        assert probes[2]["expected_decision"] == "cloud_standard"

    def test_grouped_tags_inherited(self, tmp_probes_grouped):
        from tuning.probes import load_probes

        probes = load_probes(tmp_probes_grouped)
        assert "privacy" in probes[0]["tags"]
        assert "baseline" in probes[2]["tags"]


# ---------------------------------------------------------------------------
# Engine: DSL config loading
# ---------------------------------------------------------------------------


class TestDSLConfigLoading:
    def test_load_scores(self, dsl_config):
        assert len(dsl_config.scores) == 1
        assert dsl_config.scores[0].name == "privacy_risk_score"
        assert len(dsl_config.scores[0].inputs) == 2

    def test_load_mappings(self, dsl_config):
        assert len(dsl_config.mappings) == 1
        m = dsl_config.mappings[0]
        assert m.name == "privacy_policy_band"
        assert m.source_score == "privacy_risk_score"
        assert len(m.outputs) == 2

    def test_find_score(self, dsl_config):
        assert dsl_config.find_score("privacy_risk_score") is not None
        assert dsl_config.find_score("nonexistent") is None

    def test_find_mapping_for_projection(self, dsl_config):
        mapping, band = dsl_config.find_mapping_for_projection("policy_privacy_local")
        assert mapping is not None
        assert band["gte"] == 0.35

        mapping, band = dsl_config.find_mapping_for_projection("nonexistent")
        assert mapping is None

    def test_empty_config(self):
        from tuning.engine import load_dsl_config

        dsl = load_dsl_config({})
        assert dsl.scores == []
        assert dsl.mappings == []
        assert dsl.decisions == []


# ---------------------------------------------------------------------------
# Engine: trace tree walking
# ---------------------------------------------------------------------------


class TestTraceTreeWalking:
    def test_find_failing_leaves_single(self):
        from tuning.engine import find_failing_leaves

        tree = {
            "node_type": "leaf",
            "signal_type": "projection",
            "signal_name": "policy_privacy_local",
            "matched": False,
            "confidence": 0.2,
        }
        leaves = find_failing_leaves(tree)
        assert len(leaves) == 1
        assert leaves[0].signal_name == "policy_privacy_local"
        assert leaves[0].matched is False

    def test_find_failing_leaves_and_node(self):
        from tuning.engine import find_failing_leaves

        tree = {
            "node_type": "AND",
            "matched": False,
            "children": [
                {
                    "node_type": "leaf",
                    "signal_type": "pii",
                    "signal_name": "ssn",
                    "matched": True,
                    "confidence": 0.9,
                },
                {
                    "node_type": "leaf",
                    "signal_type": "keyword",
                    "signal_name": "internal",
                    "matched": False,
                    "confidence": 0.1,
                },
            ],
        }
        leaves = find_failing_leaves(tree)
        assert len(leaves) == 1
        assert leaves[0].signal_name == "internal"

    def test_find_failing_leaves_or_picks_highest_confidence(self):
        from tuning.engine import find_failing_leaves

        tree = {
            "node_type": "OR",
            "matched": False,
            "children": [
                {
                    "node_type": "leaf",
                    "signal_type": "pii",
                    "signal_name": "ssn",
                    "matched": False,
                    "confidence": 0.1,
                },
                {
                    "node_type": "leaf",
                    "signal_type": "keyword",
                    "signal_name": "internal",
                    "matched": False,
                    "confidence": 0.8,
                },
            ],
        }
        leaves = find_failing_leaves(tree)
        assert len(leaves) == 1
        assert leaves[0].signal_name == "internal"

    def test_find_false_positive_leaves_or(self):
        from tuning.engine import find_false_positive_leaves

        tree = {
            "node_type": "OR",
            "matched": True,
            "children": [
                {
                    "node_type": "leaf",
                    "signal_type": "category_kb",
                    "signal_name": "__best__:math",
                    "matched": True,
                    "confidence": 0.7,
                },
                {
                    "node_type": "leaf",
                    "signal_type": "category_kb",
                    "signal_name": "__best__:bio",
                    "matched": False,
                    "confidence": 0.1,
                },
            ],
        }
        leaves = find_false_positive_leaves(tree)
        assert len(leaves) == 1
        assert leaves[0].signal_name == "__best__:math"

    def test_empty_tree(self):
        from tuning.engine import find_failing_leaves, find_false_positive_leaves

        assert find_failing_leaves({}) == []
        assert find_false_positive_leaves({}) == []


# ---------------------------------------------------------------------------
# Engine: score computation
# ---------------------------------------------------------------------------


class TestScoreComputation:
    def test_compute_score_weighted_sum(self, dsl_config):
        from tuning.engine import compute_score

        formula = dsl_config.scores[0]
        sc = {"pii:pii_detected": 0.9, "keyword:internal_doc": 0.7}
        ms = {"pii": ["pii_detected"], "keywords": ["internal_doc"]}
        score = compute_score(formula, sc, ms)
        expected = 0.6 * 1.0 + 0.4 * 0.7  # pii match=1.0, keyword confidence=0.7
        assert abs(score - expected) < 1e-6

    def test_compute_score_unmatched(self, dsl_config):
        from tuning.engine import compute_score

        formula = dsl_config.scores[0]
        score = compute_score(formula, {}, {})
        assert score == 0.0

    def test_compute_threshold_fix(self):
        from tuning.engine import (
            FailureClassification,
            TraceLeaf,
            compute_threshold_fix,
        )

        leaf = TraceLeaf("projection", "policy_privacy_local", False, 0.2, "root")
        fc = FailureClassification(
            kind="parametric",
            leaf=leaf,
            score_name="privacy_risk_score",
            mapping_name="privacy_policy_band",
            threshold=0.35,
            threshold_dir="gte",
            current_score=0.20,
            gap=0.15,
        )
        fix = compute_threshold_fix(fc)
        assert fix is not None
        assert fix.fix_type == "threshold"
        assert fix.new_value < 0.35
        assert fix.new_value == pytest.approx(0.18, abs=0.01)

    def test_compute_threshold_fix_returns_none_for_structural(self):
        from tuning.engine import (
            FailureClassification,
            TraceLeaf,
            compute_threshold_fix,
        )

        leaf = TraceLeaf("keyword", "marker", False, 0.0, "root")
        fc = FailureClassification(kind="structural", leaf=leaf)
        assert compute_threshold_fix(fc) is None


# ---------------------------------------------------------------------------
# Engine: config mutation
# ---------------------------------------------------------------------------


class TestConfigMutation:
    def test_apply_threshold_fix(self, dsl_config):
        from tuning.engine import Fix, apply_fix_to_config

        fix = Fix(
            fix_type="threshold",
            target="privacy_policy_band",
            param_path="projections.mappings[privacy_policy_band].outputs[policy_privacy_local].gte",
            old_value=0.35,
            new_value=0.19,
        )
        new_cfg = apply_fix_to_config(SAMPLE_DSL_CONFIG, fix, dsl_config)
        mappings = new_cfg["routing"]["projections"]["mappings"]
        m = next(m for m in mappings if m["name"] == "privacy_policy_band")
        gte_out = next(o for o in m["outputs"] if "gte" in o)
        lt_out = next(o for o in m["outputs"] if "lt" in o)
        assert gte_out["gte"] == 0.19
        assert lt_out["lt"] == 0.19  # partition consistency

    def test_apply_structural_fix(self):
        from tuning.engine import StructuralFix, apply_structural_fix

        cfg = {
            "routing": {
                "decisions": [
                    {
                        "name": "escalate_72b",
                        "rules": {
                            "operator": "OR",
                            "conditions": [
                                {"type": "category_kb", "name": "__best__:math"},
                                {"type": "category_kb", "name": "__best__:health"},
                                {"type": "category_kb", "name": "__best__:bio"},
                            ],
                        },
                    }
                ]
            }
        }
        sfix = StructuralFix(
            decision_name="escalate_72b",
            action="remove_false_positive_branches",
            description="Remove health",
            remove_signals=[{"type": "category_kb", "name": "__best__:health"}],
        )
        new_cfg = apply_structural_fix(cfg, sfix)
        conditions = new_cfg["routing"]["decisions"][0]["rules"]["conditions"]
        names = [c["name"] for c in conditions]
        assert "__best__:health" not in names
        assert "__best__:math" in names
        assert "__best__:bio" in names

    def test_apply_fix_preserves_original(self, dsl_config):
        import copy

        from tuning.engine import Fix, apply_fix_to_config

        original = copy.deepcopy(SAMPLE_DSL_CONFIG)
        fix = Fix(
            fix_type="threshold",
            target="privacy_policy_band",
            param_path="...",
            old_value=0.35,
            new_value=0.19,
        )
        apply_fix_to_config(SAMPLE_DSL_CONFIG, fix, dsl_config)
        assert original == SAMPLE_DSL_CONFIG


# ---------------------------------------------------------------------------
# Engine: severity helpers
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_probe_severity_security(self):
        from tuning.engine import probe_severity

        assert probe_severity({"tags": ["security"], "expected": "x"}) == 10
        assert probe_severity({"tags": ["jailbreak"], "expected": "x"}) == 10

    def test_probe_severity_from_expected(self):
        from tuning.engine import probe_severity

        assert (
            probe_severity({"tags": [], "expected": "local_security_containment"}) == 10
        )
        assert probe_severity({"tags": [], "expected": "local_privacy_policy"}) == 10

    def test_probe_severity_baseline(self):
        from tuning.engine import probe_severity

        assert probe_severity({"tags": ["baseline"], "expected": "x"}) == 1

    def test_probe_severity_default(self):
        from tuning.engine import probe_severity

        assert probe_severity({"tags": [], "expected": "cloud_standard"}) == 3


# ---------------------------------------------------------------------------
# Offline analyzer
# ---------------------------------------------------------------------------


class TestOfflineAnalyzer:
    @dataclass
    class Q:
        qid: str
        confidence: float
        correct_small: bool
        correct_large: bool
        quadrant: str
        category: str = ""

    def test_find_optimal_threshold_empty(self):
        from tuning.analyzer import OfflineAnalyzer

        analyzer = OfflineAnalyzer(severity_fn=lambda q: 1)
        result = analyzer.find_optimal_threshold(
            items=[],
            confidence_fn=lambda q: q.confidence,
            quadrant_fn=lambda q: q.quadrant,
            correct_small_fn=lambda q: q.correct_small,
            correct_large_fn=lambda q: q.correct_large,
            id_fn=lambda q: q.qid,
        )
        assert result["strategy"] == "AVOID"

    def test_find_optimal_threshold_all_uplift(self):
        from tuning.analyzer import OfflineAnalyzer

        items = [
            self.Q("q1", 0.3, False, True, "uplift"),
            self.Q("q2", 0.5, False, True, "uplift"),
            self.Q("q3", 0.8, False, True, "uplift"),
        ]
        analyzer = OfflineAnalyzer(severity_fn=lambda q: 1)
        result = analyzer.find_optimal_threshold(
            items=items,
            confidence_fn=lambda q: q.confidence,
            quadrant_fn=lambda q: q.quadrant,
            correct_small_fn=lambda q: q.correct_small,
            correct_large_fn=lambda q: q.correct_large,
            id_fn=lambda q: q.qid,
        )
        assert result["strategy"] == "ESCALATE"
        assert result["best_accuracy"] == 3
        assert result["escalation_rate"] > 85.0

    def test_find_optimal_threshold_mixed(self):
        from tuning.analyzer import OfflineAnalyzer

        items = [
            self.Q("q1", 0.3, False, True, "uplift"),
            self.Q("q2", 0.7, True, True, "both_correct"),
            self.Q("q3", 0.9, True, False, "regression"),
        ]
        analyzer = OfflineAnalyzer(severity_fn=lambda q: 1)
        result = analyzer.find_optimal_threshold(
            items=items,
            confidence_fn=lambda q: q.confidence,
            quadrant_fn=lambda q: q.quadrant,
            correct_small_fn=lambda q: q.correct_small,
            correct_large_fn=lambda q: q.correct_large,
            id_fn=lambda q: q.qid,
        )
        assert result["best_accuracy"] >= 2
        assert result["strategy"] in ("ESCALATE", "SELECTIVE", "AVOID")

    def test_compute_threshold_fix_regression(self):
        from tuning.analyzer import OfflineAnalyzer

        # Raising threshold from 0.3 to 0.5 escalates q1 (0.4 < 0.5).
        # q1 is an "uplift" (wrong small, correct large) → gain.
        # q2 at 0.35 is a "regression" (correct small, wrong large) → loss.
        items = [
            self.Q("q1", 0.4, False, True, "uplift"),
            self.Q("q2", 0.35, True, False, "regression"),
        ]
        analyzer = OfflineAnalyzer(severity_fn=lambda q: 1)
        fix = analyzer.compute_threshold_fix(
            items=items,
            current_threshold=0.3,
            target_threshold=0.5,
            confidence_fn=lambda q: q.confidence,
            quadrant_fn=lambda q: q.quadrant,
            id_fn=lambda q: q.qid,
        )
        assert "q1" in fix.affected
        assert "q2" in fix.affected
        assert fix.severity_gain > 0
        assert fix.severity_loss > 0


# ---------------------------------------------------------------------------
# CLI scenario discovery
# ---------------------------------------------------------------------------


class TestCLIDiscovery:
    def test_builtin_scenarios_listed(self):
        from tuning.cli import BUILTIN_SCENARIOS

        assert "privacy" in BUILTIN_SCENARIOS
        assert "calibration" in BUILTIN_SCENARIOS

    def test_load_builtin_privacy(self):
        from tuning.cli import _load_scenario

        s = _load_scenario("privacy")
        assert s.name == "privacy_routing_tuning"

    def test_load_builtin_calibration(self):
        from tuning.cli import _load_scenario

        s = _load_scenario("calibration")
        assert s.name == "calibration_tuning"

    def test_load_by_module_path(self):
        from tuning.cli import _load_scenario

        s = _load_scenario("tuning.scenarios.privacy:PrivacyScenario")
        assert s.name == "privacy_routing_tuning"

    def test_unknown_scenario_exits(self):
        from tuning.cli import _load_scenario

        with pytest.raises(SystemExit):
            _load_scenario("nonexistent_scenario")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_save_creates_file(self, tmp_path):
        from tuning.probes import save_results

        output = {"scenario": "test", "accuracy": 0.95}
        path = save_results(output, "test.json", tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["scenario"] == "test"

    def test_save_creates_dir(self, tmp_path):
        from tuning.probes import save_results

        out_dir = tmp_path / "nested" / "results"
        path = save_results({"ok": True}, "out.json", out_dir)
        assert path.exists()
        assert out_dir.is_dir()
