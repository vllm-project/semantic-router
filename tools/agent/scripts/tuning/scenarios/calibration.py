"""Calibration Scenario — domain-conditional model escalation tuning.

Starting from an "always escalate to 72B" config, tunes toward a Pareto-optimal
policy where only beneficial categories escalate.  Severity weights reflect
per-category net uplift from escalation.

Usage:
    from tuning import RouterClient, TuningLoop
    from tuning.scenarios import CalibrationScenario

    scenario = CalibrationScenario()
    loop = TuningLoop(
        scenario=scenario,
        router=RouterClient(),
        config_path=Path("config.yaml"),
        probes_path=Path("probes.yaml"),
        max_iterations=15,
    )
    output = loop.run()
"""

from __future__ import annotations

from typing import Any

from ..scenario import Scenario

NET_UPLIFTS = {
    "computer_science": 8,
    "other": 8,
    "psychology": 7,
    "biology": 6,
    "math": 5,
    "business": 3,
    "philosophy": 3,
    "economics": 2,
    "engineering": 2,
    "law": 2,
    "chemistry": 1,
    "physics": 1,
    "history": 0,
    "health": -2,
}


class CalibrationScenario(Scenario):

    @property
    def name(self) -> str:
        return "calibration_tuning"

    def severity(self, probe: dict) -> int:
        tags = probe.get("tags", [])
        cat = next((t for t in tags if t in NET_UPLIFTS), None)
        if cat is None:
            return 3
        net = abs(NET_UPLIFTS.get(cat, 0))
        if net >= 5:
            return 10
        if net >= 3:
            return 5
        if net >= 1:
            return 3
        return 1

    def adapt_result(self, probe: dict, resp: dict) -> dict | None:
        """Treat NONE as keep_7b when that's the expected decision."""
        dr = resp.get("decision_result", {})
        actual = dr.get("decision_name", "NONE")
        expected = probe["expected_decision"]

        if actual == "NONE" and expected == "keep_7b":
            correct = True
            actual = "keep_7b"
        else:
            correct = actual == expected

        return {
            "id": probe["id"],
            "query": probe["query"][:200],
            "expected": expected,
            "actual": actual,
            "correct": correct,
            "signal_confidences": resp.get("signal_confidences", {}),
            "projection_scores": resp.get("projection_scores", {}),
            "projection_bands": resp.get("projection_bands", {}),
            "eval_trace": dr.get("eval_trace", []),
            "matched_signals": dr.get("matched_signals", {}),
            "unmatched_signals": dr.get("unmatched_signals", {}),
            "tags": probe.get("tags", []),
        }

    def display_iteration(
        self,
        iteration: int,
        results: list[dict],
        diagnoses: list[dict],
        fix: Any,
    ) -> None:
        if fix is not None:
            return
        structural = sum(
            1 for d in diagnoses if "structural" in d.get("failure_kind", "")
        )
        parametric = sum(
            1 for d in diagnoses if "parametric" in d.get("failure_kind", "")
        )
        conflict = sum(
            1 for d in diagnoses if "priority_conflict" in d.get("failure_kind", "")
        )
        print(f"\n  Diagnosis summary: {len(diagnoses)} failures")
        print(
            f"    structural: {structural}, parametric: {parametric}, "
            f"priority_conflict: {conflict}"
        )
