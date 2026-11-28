"""Shared synthetic fixtures for recipe-lab tests (no network, no stack)."""

from __future__ import annotations

import json

from recipe_lab.contract import OraclePolicy

POLICY = OraclePolicy.from_dict(
    {
        "math": "local/model-math",
        "code": "local/model-code",
        "reasoning": "local/model-large",
    }
)

# F2 multi-signal oracle: first matching rule wins, default covers the rest.
RULES_POLICY = OraclePolicy.from_dict(
    {
        "rules": [
            {
                "when": {"domain": "code", "difficulty": {"gte": 3}},
                "expect": {
                    "decision": "domain_code_complex",
                    "models": ["m/pro", "m/max"],
                },
            },
            {
                "when": {"domain": "code"},
                "expect": {"decision": "domain_code", "models": ["m/code"]},
            },
        ],
        "default": {"decision": "simple_general", "models": ["m/general"]},
    }
)

VALID_SPEC: dict = {
    "api_version": "recipe-lab/v1alpha1",
    "objective": {
        "metric_key": "capability.score",
        "direction": "maximize",
        "min_improvement_pct": 1.0,
        "replay_metric": "routed_accuracy",
    },
    "workload": {
        "manifest": "datasets/dev-mmlu-pro.yaml",
        "content_hash": "sha256:" + "0" * 64,
        "profile": "quick",
    },
    "attribution": {
        "oracle_policy": {"math": "local/model-math", "code": "local/model-code"},
    },
    "gates": [
        {"name": "pii-recall-floor", "metric": "guard.pii.recall", "floor": "baseline"},
    ],
    "budget": {"rounds": 5, "max_minutes_per_trial": 60},
    "stack": {
        "mode": "external",
        "endpoint": "http://127.0.0.1:8899",
        "baseline_ref": "config-revision@main",
    },
}

# B: the preferred three-way workload form (splits must be disjoint)
VALID_SPLITS_SPEC: dict = {
    **VALID_SPEC,
    "workload": {
        "profile": "standard",
        "split_rule": "group by source_id; seed 42; 40/30/30",
        "splits": {
            "calib": {"manifest": "calib.jsonl", "content_hash": "sha256:" + "1" * 64},
            "dev": {"manifest": "dev.jsonl", "content_hash": "sha256:" + "2" * 64},
            "frozen_test": {
                "manifest": "frozen_test.jsonl",
                "content_hash": "sha256:" + "3" * 64,
            },
        },
    },
}

# M5: the multi-workload form, shaped like the V3 cold-start acceptance
# contract - tune workloads (MMLU-Pro three-way, office mcq) plus transfer
# workloads (model_eval driven, final verify only).
VALID_WORKLOADS_SPEC: dict = {
    **{k: v for k, v in VALID_SPLITS_SPEC.items() if k != "workload"},
    "workloads": [
        {
            "name": "mmlu-pro",
            "kind": "mcq",
            "weight": 2.0,
            "profile": "standard",
            "split_rule": "group by source_id; seed 42; 40/30/30",
            "splits": {
                "calib": {
                    "manifest": "calib.jsonl",
                    "content_hash": "sha256:" + "1" * 64,
                },
                "dev": {"manifest": "dev.jsonl", "content_hash": "sha256:" + "2" * 64},
                "frozen_test": {
                    "manifest": "frozen_test.jsonl",
                    "content_hash": "sha256:" + "3" * 64,
                },
            },
        },
        {
            "name": "office",
            "kind": "mcq",
            "driver": "model_eval",
            "splits": {
                "dev": {
                    "manifest": "office.jsonl",
                    "content_hash": "sha256:" + "4" * 64,
                }
            },
        },
        {
            "name": "gpqa-transfer",
            "kind": "mcq",
            "driver": "model_eval",
            "role": "transfer",
            "splits": {
                "eval": {
                    "manifest": "gpqa.jsonl",
                    "content_hash": "sha256:" + "5" * 64,
                }
            },
        },
    ],
}


def make_case(
    case_id: str, truth: str, matched_decision: str, routed: str, answer_ok, family=None
) -> dict:
    return {
        "case_id": case_id,
        "truth": truth,
        "matched_decision": matched_decision,
        "routed": routed,
        "answer_ok": answer_ok,
        **({"family": family} if family is not None else {}),
    }


def make_signal_case(
    case_id: str,
    truth_signals: dict,
    matched_decision: str,
    routed: str,
    answer_ok,
    signal_values: dict | None = None,
    signal_confidences: dict | None = None,
) -> dict:
    """F2 case: object truth with signal facts plus replay signal evidence."""
    return {
        "case_id": case_id,
        "truth": {"signals": truth_signals},
        "matched_decision": matched_decision,
        "routed": routed,
        "answer_ok": answer_ok,
        **({"signal_values": signal_values} if signal_values is not None else {}),
        **(
            {"signal_confidences": signal_confidences}
            if signal_confidences is not None
            else {}
        ),
    }


ATTRIBUTION_CASES: list[dict] = [
    # decision mismatch: expected math but the decision matched code (F1)
    make_case("case-mm", "math", "code", "local/model-code", False, family="d3"),
    make_case("case-e2", "math", "math", "local/model-code", False, family="d3"),
    make_case("case-e3", "math", "math", "local/model-math", False, family="d3"),
    make_case("case-sub", "math", "math", "local/model-code", True, family="d3"),
    make_case("case-correct", "math", "math", "local/model-math", True, family="d3"),
    make_case("case-noev", "math", "math", "local/model-math", None, family="d3"),
    make_case("case-gap", "unknown-label", "unknown-label", "local/model-math", True),
]


# ---- sub-issue 7: BUG-3 pilot-shaped diagnostic rows (D) ----


def pilot_difficulty_values() -> list[float]:
    """344 ascending difficulty scores shaped like the BUG-3 pilot (appendix
    A of sub-issue-7): range [0.08, 1.87], 74 values below 1, nothing at or
    above 2.5, and p75 anchored exactly at 1.75 so the calibrated proposal
    matches the PR #5568 golden result deterministically."""
    values = [round(0.08 + i * (0.91 / 73), 4) for i in range(74)]
    values += [round(1.0 + i * (0.74 / 182), 4) for i in range(183)]
    values += [1.75, 1.75]  # indices 257/258: the p75 interpolation anchors
    values += [round(1.7501 + i * ((1.87 - 1.7501) / 84), 4) for i in range(85)]
    assert len(values) == 344
    return sorted(values)


def pilot_rows() -> list[dict]:
    """Unlabeled diagnostic rows for the agent-recipe fixture: the strong
    lanes (domain_code_complex / complex_general) never fire, every row
    carries a difficulty score plus a per-choice domain confidence."""
    decisions = [
        "simple_math_fast_path",
        "domain_code",
        "medium_general",
        "simple_general",
    ]
    rows = []
    for index, value in enumerate(pilot_difficulty_values()):
        rows.append(
            {
                "request_id": f"req-{index:04d}",
                "matched_decision": decisions[index % 4],
                "signal_values": {
                    "decision:difficulty": value,
                    "decision:domain:code": round(0.80 + (index % 20) * 0.01, 4),
                },
                "signal_confidences": {},
            }
        )
    return rows


def pilot_rows_jsonl() -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pilot_rows())


# F2 cases over RULES_POLICY; truth facts vs observed signal evidence.
# Choice-type decision evidence stores one probability per choice
# (decision:<name>:<choice>); score-type evidence keeps the numeric score
# (decision:<name>).
SIGNAL_CASES: list[dict] = [
    # choice-signal E1: truth says physics but argmax choice is chemistry
    make_signal_case(
        "sig-choice-e1",
        {"domain": "physics", "difficulty": 3},
        "domain_chemistry",
        "m/chem",
        False,
        signal_values={
            "decision:domain:physics": 0.3,
            "decision:domain:chemistry": 0.7,
            "decision:difficulty": 3.0,
        },
    ),
    # score-signal E1: truth says difficulty 3 but the score reads 2.4
    make_signal_case(
        "sig-score-e1",
        {"domain": "code", "difficulty": 3},
        "domain_code_complex",
        "m/pro",
        False,
        signal_values={
            "decision:domain:code": 0.9,
            "decision:difficulty": 2.4,
        },
    ),
    # first rule wins (code + difficulty 4 >= 3 -> complex) but the routed
    # model is outside the expected model set
    make_signal_case(
        "sig-e2",
        {"domain": "code", "difficulty": 4},
        "domain_code_complex",
        "m/code",
        False,
        signal_values={"decision:domain:code": 0.9, "decision:difficulty": 4.0},
    ),
    # same routing deviation with a still-correct answer -> suboptimal
    make_signal_case(
        "sig-sub",
        {"domain": "code", "difficulty": 4},
        "domain_code_complex",
        "m/code",
        True,
        signal_values={"decision:domain:code": 0.9, "decision:difficulty": 4.0},
    ),
    # no rule matches (math) -> default; verified signals + as-expected route
    make_signal_case(
        "sig-default-e3",
        {"domain": "math"},
        "simple_general",
        "m/general",
        False,
        signal_values={"decision:domain:math": 0.95, "decision:domain:code": 0.05},
    ),
    # signal truth but the record carries no signal evidence at all: the
    # decision miss stays decision_mismatch (F1 fallback)
    make_signal_case(
        "sig-noev-fallback",
        {"domain": "code", "difficulty": 3},
        "domain_code",
        "m/code",
        False,
    ),
    # signals verified but the decision table still missed -> E2 policy error
    make_signal_case(
        "sig-verified-e2",
        {"domain": "code", "difficulty": 3},
        "domain_code",
        "m/code",
        False,
        signal_values={"decision:domain:code": 0.9, "decision:difficulty": 3.0},
    ),
]
