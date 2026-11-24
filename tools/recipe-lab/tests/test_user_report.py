"""Exercise the scope and disclosure boundary of the human report."""

import pytest
from recipe_lab.user_report import render_report

CONFIG = "4" * 64
DRAFT = {
    "schema": "recipe-lab-v05-speed-draft-v1",
    "status": "statically_validated",
    "template": {"bundle": "mom-v1", "recipe": "speed"},
    "public_model": "lab/router",
    "router_warning_count": 1,
    "bindings": {
        "fast": [{"model": "tiny"}, {"model": "large"}],
        "reasoning": [{"model": "large"}],
        "tools": [{"model": "large"}],
    },
    "sha256": {"config.yaml": CONFIG},
    "gateway_started": False,
    "business_quality_measured": False,
}
RUN = {
    "schema": "recipe-lab-sr-bench-run-v1",
    "baseline_run": "r1",
    "candidate_run": "r2",
    "baseline_target": "parent",
    "candidate_target": "candidate",
    "candidate_sha256": CONFIG,
    "min_speedup_pct": 15,
    "verdict": "FAIL",
    "reason": "protected case lost or task success fell",
    "metrics": {
        "paired_cases": 4,
        "common_successes": 2,
        "baseline_correct": 3,
        "candidate_correct": 2,
        "lost_cases": ["hidden-task"],
        "guard_losses": ["hidden-task"],
        "baseline_common_success_s": 100.0,
        "candidate_common_success_s": 75.0,
        "speedup_pct": 25.0,
    },
}

DEPTH = {
    "schema": "recipe-lab-sr-bench-depth-v1",
    "status": "proposed",
    "dev_run": "first",
    "parent_target": "parent",
    "fast_target": "fast",
    "signal": "depth",
    "before": 0.4,
    "after": 0.65,
    "candidate_sha256": CONFIG,
    "observations": [{"role": "move", "case": "hidden-task"}],
}


def test_bench_report_keeps_order_limits_quality_and_provenance():
    awaiting = {
        **RUN,
        "verdict": "INCONCLUSIVE",
        "candidate_run": None,
        "metrics": None,
        "reason": "candidate submitted; awaiting evidence",
    }
    report = render_report([awaiting, RUN], background="Tune the service.")
    for substring in (
        "Tune the service.",
        CONFIG,
        "`parent`",
        "3/4",
        "2/4",
        "2 joint successes",
        "100 s",
        "75 s",
        "25%",
        "FAIL",
        "1 on protected cases",
    ):
        assert substring in report
    assert "hidden-task" not in report
    assert (
        report.index("Latest supplied result")
        < report.index("Receipt 2. FAIL")
        < report.index("Recorded results")
    )
    assert report.count("FAIL. The candidate missed") == 1


def test_service_target_names_and_generated_background():
    report = render_report([{**RUN, "candidate_target": "decision:local/fast"}])
    assert "`decision:local/fast`" in report
    assert "Check whether the candidate preserves graded task success" in report


def test_draft_receipt_ranks_before_depth_and_grade():
    report = render_report([DRAFT, DEPTH, RUN], background="Tune the service.")
    assert (
        report.index("Generate a recipe")
        < report.index("Propose a depth change")
        < report.index("Grade a benchmark")
    )
    for substring in (
        "static configuration validation",
        "config.yaml",
        CONFIG,
        "`tiny` then `large`",
        "Router warning count was 1",
    ):
        assert substring in report
    # the draft never claims measured gateway behavior or task quality
    assert "unmeasured by this receipt" in report


def test_depth_receipt_ranks_between_generation_and_grade():
    report = render_report([DEPTH, RUN], background="Tune the service.")
    assert report.index("Propose a depth change") < report.index("Grade a benchmark")
    for substring in (
        "run `first`",
        "1 possible moves, 0 protected cases and 0 skipped cases",
        "`depth` threshold changes from 0.4 to 0.65",
        CONFIG,
    ):
        assert substring in report
    assert "hidden-task" not in report


def test_depth_keep_parent_and_inconclusive_outcomes():
    keep = {
        **DEPTH,
        "status": "keep-parent",
        "dev_run": "second",
        "reason": "no separating boundary",
    }
    del keep["before"], keep["after"], keep["candidate_sha256"], keep["observations"]
    report = render_report([keep])
    assert (
        report.count(
            "Keep the parent. The observed depth scores offered no separating threshold."
        )
        == 1
    )
    inconclusive = {**keep, "status": "inconclusive"}
    report = render_report([inconclusive])
    assert "The depth proposal is inconclusive." in report


def test_inconclusive_error_does_not_export_private_fields_or_errors():
    attempt = {
        **RUN,
        "verdict": "INCONCLUSIVE",
        "candidate_run": None,
        "baseline_target": "https://user:private@host/path",
        "metrics": None,
        "reason": "private traceback with API token",
        "trace": {"content": "hidden request"},
    }
    report = render_report(
        [attempt], background="Goal\n## note\napi_key=sk-hidden-example"
    )
    assert "INCONCLUSIVE" in report and "no run ID recorded" in report
    assert "    ## note" in report
    assert "[redacted]" in report
    for private in ("private", "hidden request", "sk-hidden-example", "://"):
        assert private not in report


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [{}],
        [{"schema": "other"}],
        [{"kind": "measurement", "split": "holdout"}],
        [{**RUN, "verdict": "PASS", "metrics": None}],
        [{**RUN, "candidate_sha256": "../bad"}],
        [{**DRAFT, "sha256": {"config.yaml": "../bad"}}],
        [{**DRAFT, "bindings": {"fast": [], "reasoning": [], "tools": []}}],
    ],
)
def test_rejects_empty_unknown_and_malformed_evidence(evidence):
    with pytest.raises(ValueError):
        render_report(evidence)
