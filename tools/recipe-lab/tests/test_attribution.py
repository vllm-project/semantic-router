"""Attribution tests: the E1/E2/E3 taxonomy and the evidence-first rule."""

from __future__ import annotations

import json

import pytest
from recipe_lab.attribution import (
    CORRECT,
    DECISION_MISMATCH,
    E1_SIGNAL,
    E2_POLICY,
    E3_CAPABILITY,
    NO_EVIDENCE,
    POLICY_GAP,
    SUBOPTIMAL,
    CaseInput,
    attribute,
    classify_case,
    load_records,
    render_markdown,
    summarize,
    write_jsonl,
)
from recipe_lab.contract import ContractError, OraclePolicy

from tests.fixtures import (
    ATTRIBUTION_CASES,
    POLICY,
    RULES_POLICY,
    SIGNAL_CASES,
    make_case,
    make_signal_case,
)


def test_all_classes_classified():
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
        dataset_hash="sha256:" + "2" * 64,
    )
    by_id = {r["case_id"]: r for r in rows}
    # string-only truth: a mismatched decision is decision_mismatch, not E1 (F1)
    assert by_id["case-mm"]["error_class"] == DECISION_MISMATCH
    assert by_id["case-e2"]["error_class"] == E2_POLICY
    assert by_id["case-e3"]["error_class"] == E3_CAPABILITY
    assert by_id["case-sub"]["error_class"] == SUBOPTIMAL
    assert by_id["case-correct"]["error_class"] == CORRECT
    assert by_id["case-noev"]["error_class"] == NO_EVIDENCE
    assert by_id["case-gap"]["error_class"] == POLICY_GAP
    # dataset hash is stamped on every row
    assert all(r["dataset_hash"] == "sha256:" + "2" * 64 for r in rows)


def test_e1_never_produced_from_string_truth():
    # F1: without signal-level truth, no record can be classified as E1
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
    )
    assert all(r["error_class"] != E1_SIGNAL for r in rows)
    assert rows[0]["matched_decision"] == "code"  # renamed row field is present


def test_legacy_pred_field_still_readable():
    # records written before F1 use "pred"; they must keep loading
    row = classify_case(
        {
            "case_id": "x",
            "truth": "math",
            "pred": "code",
            "routed": "m",
            "answer_ok": False,
        },
        POLICY,
    )
    assert row["error_class"] == DECISION_MISMATCH
    assert row["matched_decision"] == "code"


def test_evidence_first_no_evidence_with_deviation():
    # answer unknown + routing deviates -> NO_EVIDENCE (never counted as an error)
    row = classify_case(
        make_case("x", "math", "math", "local/model-code", None), POLICY
    )
    assert row["error_class"] == NO_EVIDENCE


def test_policy_gap_precedes_everything():
    row = classify_case(
        make_case("x", "no-such-label", "also-wrong", "m", None), POLICY
    )
    assert row["error_class"] == POLICY_GAP
    assert row["expected"] is None


def test_summary_math():
    summary = summarize(
        attribute(
            [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
            POLICY,
        )
    )
    assert summary["total"] == 7
    assert summary["answered"] == 4  # correct + e2 + e3 + suboptimal
    assert summary["correct"] == 1
    assert summary["accuracy"] == pytest.approx(0.25)
    assert summary["class_counts"][DECISION_MISMATCH] == 1
    assert summary["class_counts"][E1_SIGNAL] == 0  # no signal truth in these cases
    # decision mismatch pair: expected math but the decision matched code
    assert {
        "expected_decision": "math",
        "matched_decision": "code",
        "count": 1,
    } in summary["decision_mismatch_top"]
    assert summary["e1_confusion_top"] == []
    # E2 deviation: routed code, expected model-math
    assert {
        "routed": "local/model-code",
        "expected": "local/model-math",
        "count": 1,
    } in summary["e2_deviations_top"]
    assert summary["e1_by_family"] == {}
    assert summary["e1_by_signal"] == {}


def test_report_renders_family_aware_hints():
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
    )
    text = render_markdown(summarize(rows), dataset_hash="sha256:abc")
    assert "Misroute attribution report" in text
    assert "sha256:abc" in text
    assert "decision_mismatch confusion pairs" in text
    assert "`math` but decision matched `code`" in text
    assert "E1 by signal family" in text
    assert "E1 by failing signal" in text
    assert "- none (E1 needs signal-level truth" not in text
    assert "E2 routing deviations" in text


# --- F2: multi-signal oracle policy and true E1 -----------------------------


def signal_rows():
    return attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(SIGNAL_CASES)], RULES_POLICY
    )


def test_signal_cases_classified():
    by_id = {r["case_id"]: r for r in signal_rows()}
    # choice-signal E1: truth says physics, argmax choice is chemistry
    assert by_id["sig-choice-e1"]["error_class"] == E1_SIGNAL
    assert by_id["sig-choice-e1"]["e1_signal"] == "domain"
    assert by_id["sig-choice-e1"]["e1_observed"] == "chemistry"
    # family derived from the failing signal's evidence key (decision:...)
    assert by_id["sig-choice-e1"]["family"] == "vela_decisions"
    # score-signal E1: truth says difficulty 3, score reads 2.4
    assert by_id["sig-score-e1"]["error_class"] == E1_SIGNAL
    assert by_id["sig-score-e1"]["e1_signal"] == "difficulty"
    assert by_id["sig-score-e1"]["e1_observed"] == 2.4
    # first rule wins (code + difficulty 4 >= 3); model outside the set
    assert by_id["sig-e2"]["error_class"] == E2_POLICY
    assert by_id["sig-e2"]["expected_decision"] == "domain_code_complex"
    assert by_id["sig-e2"]["expected_models"] == ["m/pro", "m/max"]
    assert by_id["sig-sub"]["error_class"] == SUBOPTIMAL
    # no rule matches (math) -> default; verified signals; route as intended
    assert by_id["sig-default-e3"]["error_class"] == E3_CAPABILITY
    assert by_id["sig-default-e3"]["family"] == "vela_decisions"
    # signal truth without any evidence: decision miss stays decision_mismatch
    assert by_id["sig-noev-fallback"]["error_class"] == DECISION_MISMATCH
    # signals fully verified but the decision table missed -> policy error
    assert by_id["sig-verified-e2"]["error_class"] == E2_POLICY
    assert by_id["sig-verified-e2"]["expected_decision"] == "domain_code_complex"


def test_signal_case_signals_detail():
    row = classify_case(SIGNAL_CASES[0], RULES_POLICY)
    signals = {s["signal"]: s for s in row["signals"]}
    assert signals["domain"]["ok"] is False
    assert signals["domain"]["judged"] is True
    assert signals["domain"]["observed"] == "chemistry"
    assert signals["difficulty"]["ok"] is True
    assert signals["difficulty"]["observed"] == 3.0


def test_choice_signal_hit_label_is_argmax():
    row = classify_case(
        make_signal_case(
            "x",
            {"domain": "code"},
            "domain_code",
            "m/code",
            True,
            signal_values={"decision:domain:math": 0.4, "decision:domain:code": 0.6},
        ),
        RULES_POLICY,
    )
    assert row["error_class"] == CORRECT
    assert row["signals"][0]["observed"] == "code"


def test_predicate_fact_e1():
    row = classify_case(
        make_signal_case(
            "x",
            {"domain": "code", "difficulty": {"gte": 3}},
            "domain_code",
            "m/code",
            False,
            signal_values={"decision:domain:code": 0.9, "decision:difficulty": 2.5},
        ),
        RULES_POLICY,
    )
    assert row["error_class"] == E1_SIGNAL
    assert row["e1_signal"] == "difficulty"


def test_confidence_fallback_for_evidence():
    # per-key value fallback: signal_values first, then signal_confidences
    row = classify_case(
        make_signal_case(
            "x",
            {"domain": "code"},
            "domain_code",
            "m/code",
            True,
            signal_confidences={"decision:domain:code": 0.9},
        ),
        RULES_POLICY,
    )
    assert row["error_class"] == CORRECT


def test_model_set_membership_allows_any_expected_model():
    for routed in ("m/pro", "m/max"):
        row = classify_case(
            make_signal_case(
                "x",
                {"domain": "code", "difficulty": 4},
                "domain_code_complex",
                routed,
                True,
                signal_values={"decision:domain:code": 0.9, "decision:difficulty": 4.0},
            ),
            RULES_POLICY,
        )
        assert row["error_class"] == CORRECT


def test_e1_precedes_no_evidence():
    # a contradicting signal is answer-independent evidence
    row = classify_case(
        make_signal_case(
            "x",
            {"domain": "math"},
            "domain_code",
            "m/code",
            None,
            signal_values={"decision:domain:code": 0.9, "decision:domain:math": 0.1},
        ),
        RULES_POLICY,
    )
    assert row["error_class"] == E1_SIGNAL
    assert row["e1_signal"] == "domain"


def test_no_evidence_precedes_routing_verdict_with_signal_truth():
    row = classify_case(
        make_signal_case(
            "x",
            {"domain": "code"},
            "domain_code",
            "m/code",
            None,
            signal_values={"decision:domain:code": 0.9},
        ),
        RULES_POLICY,
    )
    assert row["error_class"] == NO_EVIDENCE


def test_string_truth_under_rules_policy_is_gap():
    row = classify_case(make_case("x", "math", "math", "m/general", True), RULES_POLICY)
    assert row["error_class"] == POLICY_GAP


def test_signal_truth_policy_gap_without_default():
    policy = OraclePolicy.from_dict(
        {"rules": [{"when": {"domain": "code"}, "expect": {"decision": "domain_code"}}]}
    )
    row = classify_case(
        make_signal_case("x", {"domain": "math"}, "simple_general", "m", True), policy
    )
    assert row["error_class"] == POLICY_GAP


def test_summary_includes_e1_signal_breakdown():
    summary = summarize(signal_rows())
    assert summary["class_counts"][E1_SIGNAL] == 2
    assert summary["e1_by_signal"] == {"domain": 1, "difficulty": 1}
    assert summary["e1_by_family"] == {"vela_decisions": 2}
    # choice-e1 case: facts resolve to the default expectation
    assert {
        "expected_decision": "simple_general",
        "matched_decision": "domain_chemistry",
        "count": 1,
    } in summary["e1_confusion_top"]
    assert summary["class_counts"][DECISION_MISMATCH] == 1


def test_report_renders_e1_signal_truth_sections():
    text = render_markdown(summarize(signal_rows()))
    assert "E1 by failing signal" in text
    assert "`domain`: 1 cases" in text
    assert "`difficulty`: 1 cases" in text
    assert "threshold calibration / signal-question engineering" in text
    assert "- none (E1 needs signal-level truth" not in text


@pytest.mark.parametrize(
    "truth",
    [
        {"unexpected": 1},
        {"signals": {}},
        {"signals": {"domain": None}},
        {"signals": {"domain": {"within": 3}}},
        {"signals": {"": "code"}},
        42,
    ],
)
def test_truth_validation_rejects(truth):
    with pytest.raises(ContractError, match="truth"):
        CaseInput.from_dict(
            {
                "case_id": "x",
                "truth": truth,
                "matched_decision": "d",
                "routed": "m",
                "answer_ok": True,
            },
            0,
        )


def test_signal_values_validation_rejects():
    with pytest.raises(ContractError, match="signal_values"):
        CaseInput.from_dict(
            {
                "case_id": "x",
                "truth": "math",
                "matched_decision": "math",
                "routed": "m",
                "answer_ok": True,
                "signal_values": {"decision:difficulty": "high"},
            },
            0,
        )
    with pytest.raises(ContractError, match="signal_confidences"):
        CaseInput.from_dict(
            {
                "case_id": "x",
                "truth": "math",
                "matched_decision": "math",
                "routed": "m",
                "answer_ok": True,
                "signal_confidences": {"decision:difficulty": True},
            },
            0,
        )


def test_write_jsonl_roundtrip(tmp_path):
    rows = attribute(
        [CaseInput.from_dict(c, i) for i, c in enumerate(ATTRIBUTION_CASES)],
        POLICY,
        dataset_hash="sha256:" + "3" * 64,
    )
    out = write_jsonl(rows, tmp_path / "attribution.jsonl")
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows


def test_load_records_rejects_bad_answer_ok(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "a",
                "truth": "math",
                "pred": "math",
                "routed": "m",
                "answer_ok": "yes",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="answer_ok"):
        load_records(path)


def test_load_records_rejects_missing_field(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"case_id": "a", "truth": "math"}), encoding="utf-8")
    with pytest.raises(ContractError, match="required field"):
        load_records(path)
