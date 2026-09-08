package evaluationplane

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestReduceRoutingRecipeEvaluationGolden(t *testing.T) {
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	input := RoutingRecipeReductionInput{
		Plan: testRoutingRecipePlan(), ExpectedCaseIDs: []string{"case-1", "case-2", "case-3"},
		Decisions: []RoutingRecipeDecisionSnapshot{
			testRoutingRecipeDecision("case-1", "decision-1", base, "arm-a", []string{"arm-a", "arm-b"}, 0.9, 0.8, "present"),
			testRoutingRecipeDecision("case-2", "decision-2", base.Add(time.Second), "arm-b", []string{"arm-b", "arm-a"}, 0.2, 0.4, "missing"),
			testRoutingRecipeDecision("case-3", "decision-3", base.Add(2*time.Second), "arm-a", []string{"arm-a", "arm-b"}, 0.8, 0.5, "error"),
		},
		Outcomes: []RoutingRecipeOutcome{
			{DecisionID: "decision-1", CaseID: "case-1", ArmID: "arm-a", ObservedAt: base.Add(time.Minute), Quality: 0.8},
			{DecisionID: "decision-1", CaseID: "case-1", ArmID: "arm-b", ObservedAt: base.Add(time.Minute), Quality: 0.7},
			{DecisionID: "decision-2", CaseID: "case-2", ArmID: "arm-a", ObservedAt: base.Add(time.Minute), Quality: 0.9},
			{DecisionID: "decision-2", CaseID: "case-2", ArmID: "arm-b", ObservedAt: base.Add(time.Minute), Quality: 0.4},
			{DecisionID: "decision-3", CaseID: "case-3", ArmID: "arm-a", ObservedAt: base.Add(time.Minute), Quality: 0.5},
			{DecisionID: "decision-3", CaseID: "case-3", ArmID: "arm-b", ObservedAt: base.Add(time.Minute), Quality: 0.5},
		},
	}
	report, err := ReduceRoutingRecipeEvaluation(input)
	if err != nil {
		t.Fatalf("reduce routing recipe evaluation: %v", err)
	}
	if report.ContractVersion != RoutingRecipeEvaluationContractVersion || report.E1.ExpectedDecisions != 3 || report.E1.ObservedDecisions != 3 || report.E1.SelectedFeasible != 3 {
		t.Fatalf("unexpected E1 decision summary: %+v", report.E1)
	}
	signal := inputAvailabilityReportByID(t, report.E1.Signals, "complexity:complexity")
	if signal.Present != 1 || signal.Missing != 1 || signal.Error != 1 || signal.Timeout != 0 || signal.Latency.Available || signal.Latency.Reason != "insufficient_latency_samples" {
		t.Fatalf("unexpected E1 signal availability: %+v", signal)
	}
	availability := inputAvailabilityReportByID(t, report.E1.Signals, "context:availability")
	if availability.Present != 2 || availability.Timeout != 1 || !availability.Latency.Available || availability.Latency.P50MS != 3 || availability.Latency.P95MS != 3 {
		t.Fatalf("unexpected E1 timeout/latency summary: %+v", availability)
	}
	quality := projectionReportByID(t, report, "projection:quality-score")
	if !quality.Spearman.Available || quality.Spearman.Value != 1 || quality.Brier.Available || quality.Brier.Reason != "calibration_target_not_binary" {
		t.Fatalf("unexpected quality projection report: %+v", quality)
	}
	probability := projectionReportByID(t, report, "projection:oracle-probability")
	if !probability.Spearman.Available || !probability.Brier.Available || !probability.ECE10.Available || len(probability.Reliability) != 10 {
		t.Fatalf("unexpected probability projection availability: %+v", probability)
	}
	assertRoutingRecipeClose(t, probability.Brier.Value, 0.03)
	assertRoutingRecipeClose(t, probability.ECE10.Value, 1.0/6.0)
	if probability.Reliability[2].Count != 1 || probability.Reliability[8].Count != 1 || probability.Reliability[9].Count != 1 {
		t.Fatalf("unexpected fixed calibration bins: %+v", probability.Reliability)
	}
	if got := report.E2.TopK[0]; got.K != 1 || !got.Recall.Available {
		t.Fatalf("top-k 1 unavailable: %+v", got)
	} else {
		assertRoutingRecipeClose(t, got.Recall.Value, 2.0/3.0)
	}
	if got := report.E2.TopK[1]; got.K != 2 || !got.Recall.Available {
		t.Fatalf("top-k 2 unavailable: %+v", got)
	} else {
		assertRoutingRecipeClose(t, got.Recall.Value, 1)
	}
	if !report.E2.OracleRegret.Available {
		t.Fatalf("oracle regret unavailable: %+v", report.E2.OracleRegret)
	}
	assertRoutingRecipeClose(t, report.E2.OracleRegret.Value, 1.0/6.0)

	permuted := input
	permuted.ExpectedCaseIDs = []string{"case-3", "case-1", "case-2"}
	permuted.Decisions = []RoutingRecipeDecisionSnapshot{input.Decisions[2], input.Decisions[0], input.Decisions[1]}
	permuted.Outcomes = []RoutingRecipeOutcome{input.Outcomes[5], input.Outcomes[2], input.Outcomes[4], input.Outcomes[0], input.Outcomes[3], input.Outcomes[1]}
	permutedReport, err := ReduceRoutingRecipeEvaluation(permuted)
	if err != nil {
		t.Fatalf("reduce permuted routing recipe evaluation: %v", err)
	}
	if !reflect.DeepEqual(report, permutedReport) {
		t.Fatalf("canonical reduction changed under input permutation: %#v != %#v", report, permutedReport)
	}
}

func TestReduceRoutingRecipeEvaluationFailsClosedForLeakageAndIncompletePool(t *testing.T) {
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	input := RoutingRecipeReductionInput{
		Plan: testRoutingRecipePlan(), ExpectedCaseIDs: []string{"case-1", "case-2", "case-3"},
		Decisions: []RoutingRecipeDecisionSnapshot{
			testRoutingRecipeDecision("case-1", "decision-1", base, "arm-a", []string{"arm-a", "arm-b"}, 0.9, 0.8, "present"),
			testRoutingRecipeDecision("case-2", "decision-2", base.Add(time.Second), "arm-b", []string{"arm-b", "arm-a"}, 0.2, 0.4, "present"),
			testRoutingRecipeDecision("case-3", "decision-3", base.Add(2*time.Second), "arm-a", []string{"arm-a", "arm-b"}, 0.8, 0.5, "present"),
		},
		Outcomes: []RoutingRecipeOutcome{
			{DecisionID: "decision-1", CaseID: "case-1", ArmID: "arm-a", ObservedAt: base.Add(time.Minute), Quality: 0.8},
			{DecisionID: "decision-1", CaseID: "case-1", ArmID: "arm-b", ObservedAt: base.Add(time.Minute), Quality: 0.7},
			{DecisionID: "decision-2", CaseID: "case-2", ArmID: "arm-a", ObservedAt: base.Add(time.Minute), Quality: 0.9},
			{DecisionID: "decision-2", CaseID: "case-2", ArmID: "arm-b", ObservedAt: base.Add(time.Minute), Quality: 0.4},
			{DecisionID: "decision-3", CaseID: "case-3", ArmID: "arm-a", ObservedAt: base.Add(time.Minute), Quality: 0.5},
		},
	}
	report, err := ReduceRoutingRecipeEvaluation(input)
	if err != nil {
		t.Fatalf("incomplete pool should be reportable unavailable, got %v", err)
	}
	if report.E2.TopK[0].Recall.Available || report.E2.TopK[0].Recall.Reason != "incomplete_eligible_pool" || report.E2.OracleRegret.Available {
		t.Fatalf("incomplete pool did not fail closed: %+v", report.E2)
	}
	input.Outcomes = input.Outcomes[:0]
	input.Outcomes = append(input.Outcomes, RoutingRecipeOutcome{DecisionID: "decision-1", CaseID: "case-1", ArmID: "arm-a", ObservedAt: base, Quality: 0.8})
	if _, err := ReduceRoutingRecipeEvaluation(input); err == nil || !strings.Contains(err.Error(), "precedes or equals") {
		t.Fatalf("pre-decision outcome was accepted: %v", err)
	}
}

func TestReduceRoutingRecipeE1DoesNotCallUnknownEligibilityComplete(t *testing.T) {
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	plan := testRoutingRecipePlan()
	complete := testRoutingRecipeDecision("case-1", "decision-1", base, "arm-a", []string{"arm-a", "arm-b"}, 0.9, 0.8, "present")
	unknown := testRoutingRecipeDecision("case-2", "decision-2", base.Add(time.Second), "arm-a", []string{"arm-a"}, 0.7, 0.6, "present")
	unknown.Eligibility[1] = RoutingRecipeEligibility{ArmID: "arm-b", State: "unavailable", ReasonCode: "runtime-missing"}
	if err := ValidateRoutingRecipeDecisionSnapshot(plan, unknown); err != nil {
		t.Fatalf("unknown eligibility snapshot should remain valid fail-closed evidence: %v", err)
	}

	report := reduceRoutingRecipeE1(
		plan,
		[]string{"case-1", "case-2"},
		map[string]RoutingRecipeDecisionSnapshot{"case-1": complete, "case-2": unknown},
	)

	if report.EligibilityComplete != 1 || report.SelectedFeasible != 2 {
		t.Fatalf("unknown arm eligibility was counted as complete: %+v", report)
	}
}

func projectionReportByID(t *testing.T, report RoutingRecipeEvaluationReport, id string) RoutingRecipeProjectionOutcomeReport {
	t.Helper()
	for _, candidate := range report.E2.ProjectionOutcomes {
		if candidate.ProjectionID == id {
			return candidate
		}
	}
	t.Fatalf("projection report %q not found", id)
	return RoutingRecipeProjectionOutcomeReport{}
}

func inputAvailabilityReportByID(t *testing.T, reports []RoutingRecipeInputAvailabilityReport, id string) RoutingRecipeInputAvailabilityReport {
	t.Helper()
	for _, candidate := range reports {
		if candidate.ID == id {
			return candidate
		}
	}
	t.Fatalf("input availability report %q not found", id)
	return RoutingRecipeInputAvailabilityReport{}
}

func assertRoutingRecipeClose(t *testing.T, actual, expected float64) {
	t.Helper()
	if math.Abs(actual-expected) > 1e-12 {
		t.Fatalf("got %.16f, want %.16f", actual, expected)
	}
}
