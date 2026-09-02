package evaluationplane

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestRoutingRecipePlanDigestBindsCanonicalBody(t *testing.T) {
	plan := testRoutingRecipePlan()
	permuted := plan
	permuted.ArmIDs = []string{"arm-b", "arm-a"}
	permuted.Signals = []RoutingRecipeInputSpec{plan.Signals[1], plan.Signals[0]}
	if err := ValidateRoutingRecipePlan(permuted); err != nil {
		t.Fatalf("semantic set permutation should preserve the canonical plan identity: %v", err)
	}
	mutated := plan
	mutated.FallbackArmID = "arm-a"
	if err := ValidateRoutingRecipePlan(mutated); err == nil {
		t.Fatal("changed plan body retained the old digest")
	}
	if reflect.DeepEqual(mutated, plan) {
		t.Fatal("test mutation did not alter plan")
	}
}

func TestValidateRoutingRecipeDecisionSnapshotRejectsInvalidDecisionTimeShape(t *testing.T) {
	plan := testRoutingRecipePlan()
	snapshot := testRoutingRecipeDecision("case-1", "decision-1", time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC), "arm-a", []string{"arm-a", "arm-b"}, 0.9, 0.8, "present")
	if err := ValidateRoutingRecipeDecisionSnapshot(plan, snapshot); err != nil {
		t.Fatalf("valid snapshot rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*RoutingRecipeDecisionSnapshot)
	}{
		{"ranked noneligible arm", func(candidate *RoutingRecipeDecisionSnapshot) {
			candidate.Eligibility[1].State, candidate.Eligibility[1].ReasonCode = "ineligible", "policy"
		}},
		{"selected rank is not first", func(candidate *RoutingRecipeDecisionSnapshot) { candidate.RankedArmIDs = []string{"arm-b", "arm-a"} }},
		{"fallback is not frozen", func(candidate *RoutingRecipeDecisionSnapshot) {
			candidate.SelectionStatus, candidate.SelectedArmID = "fallback", "arm-a"
		}},
		{"missing value on present projection", func(candidate *RoutingRecipeDecisionSnapshot) { candidate.Projections[0].Value = nil }},
		{"value on timeout", func(candidate *RoutingRecipeDecisionSnapshot) {
			candidate.Signals[0].State, candidate.Signals[0].Value = "timeout", floatPointer(0.1)
		}},
		{"duplicate ranking", func(candidate *RoutingRecipeDecisionSnapshot) { candidate.RankedArmIDs = []string{"arm-a", "arm-a"} }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := snapshot
			candidate.Signals = append([]RoutingRecipeObservedInput(nil), snapshot.Signals...)
			candidate.Projections = append([]RoutingRecipeObservedInput(nil), snapshot.Projections...)
			candidate.Eligibility = append([]RoutingRecipeEligibility(nil), snapshot.Eligibility...)
			candidate.RankedArmIDs = append([]string(nil), snapshot.RankedArmIDs...)
			test.mutate(&candidate)
			if err := ValidateRoutingRecipeDecisionSnapshot(plan, candidate); err == nil {
				t.Fatal("invalid snapshot was accepted")
			}
		})
	}
}

func TestRoutingRecipeMetricNamespaceGrammarIsStrict(t *testing.T) {
	if isRoutingRecipeMetricID("routing_recipe.e2.feasible_oracle_recall_at_0") ||
		isRoutingRecipeMetricID("routing_recipe.e2.projection.x.injected") ||
		!isRoutingRecipeMetricID("routing_recipe.e1.signal.626f677573.present_rate") ||
		isRoutingRecipeMetricID("routing_recipe.e1.signal.7369676e616c2e616c7068612e62657461.present_rate.injected") {
		t.Fatal("routing recipe metric namespace grammar is not strict")
	}
}

func TestRoutingRecipeInputIDsUseBoundedTypedRuntimeKeys(t *testing.T) {
	for _, test := range []struct {
		id         string
		projection bool
		valid      bool
	}{
		{id: "domain:reasoning", valid: true},
		{id: "classifier:risk:RISKY", valid: true},
		{id: "kb_metric:docs:best_score", valid: true},
		{id: "projection:quality", projection: true, valid: true},
		{id: "case-id", valid: false},
		{id: "unknown:signal", valid: false},
		{id: "DOMAIN:reasoning", valid: false},
		{id: "domain:bad/value", valid: false},
		{id: "domain:a:b", valid: false},
		{id: "classifier:risk:RISKY:extra", valid: false},
		{id: "projection:quality", valid: false},
		{id: "domain:reasoning", projection: true, valid: false},
		{id: "domain:" + strings.Repeat("a", 121), valid: true},
		{id: "domain:" + strings.Repeat("a", 122), valid: false},
	} {
		if got := validRoutingRecipeInputID(test.id, test.projection); got != test.valid {
			t.Fatalf("validRoutingRecipeInputID(%q, %v) = %v, want %v", test.id, test.projection, got, test.valid)
		}
	}
	if validRoutingRecipeID("domain:reasoning") || !validRoutingRecipeID("case-id") || !validRoutingRecipeID("arm-A") {
		t.Fatal("runtime input grammar leaked into generic case/arm ID validation")
	}
}

func testRoutingRecipePlan() RoutingRecipePlan {
	plan, err := canonicalRoutingRecipePlan(RoutingRecipePlan{
		ContractVersion:      RoutingRecipePlanContractVersion,
		TargetSnapshotDigest: digestForRoutingRecipeTest('b'),
		ArmIDs:               []string{"arm-a", "arm-b"}, FallbackArmID: "arm-b",
		Signals: []RoutingRecipeInputSpec{{ID: "complexity:complexity", ValueKind: "numeric"}, {ID: "context:availability", ValueKind: "numeric"}},
		Projections: []RoutingRecipeProjectionSpec{
			{ID: "projection:oracle-probability", ValueKind: "probability", OutcomeBinding: "selected_is_oracle"},
			{ID: "projection:quality-score", ValueKind: "numeric", OutcomeBinding: "selected_pool_quality"},
		},
		TopK: []int{1, 2},
	})
	if err != nil {
		panic(err)
	}
	return plan
}

func testRoutingRecipeDecision(caseID, decisionID string, observedAt time.Time, selected string, ranked []string, probability, quality float64, signalState string) RoutingRecipeDecisionSnapshot {
	signal := RoutingRecipeObservedInput{ID: "complexity:complexity", State: signalState, LatencyMS: floatPointer(2)}
	switch signalState {
	case "present":
		signal.Value = floatPointer(0.4)
	case "error":
		signal.ErrorCode = "upstream"
	}
	availability := RoutingRecipeObservedInput{ID: "context:availability", State: "present", Value: floatPointer(0.7), LatencyMS: floatPointer(3)}
	if signalState == "missing" {
		availability.State, availability.Value = "timeout", nil
	}
	return RoutingRecipeDecisionSnapshot{
		ContractVersion: RoutingDecisionEvidenceContractVersion, DecisionID: decisionID,
		PlanDigest: testRoutingRecipePlan().PlanDigest, CaseID: caseID, ObservedAt: observedAt,
		Signals: []RoutingRecipeObservedInput{signal, availability},
		Projections: []RoutingRecipeObservedInput{
			{ID: "projection:oracle-probability", State: "present", Value: floatPointer(probability), LatencyMS: floatPointer(1)},
			{ID: "projection:quality-score", State: "present", Value: floatPointer(quality), LatencyMS: floatPointer(2)},
		},
		Eligibility:  []RoutingRecipeEligibility{{ArmID: "arm-a", State: "eligible", ReasonCode: "none"}, {ArmID: "arm-b", State: "eligible", ReasonCode: "none"}},
		RankedArmIDs: ranked, SelectedArmID: selected, SelectionStatus: "selected",
	}
}

func digestForRoutingRecipeTest(value rune) string {
	return "sha256:" + strings.Repeat(fmt.Sprintf("%c", value), 64)
}
