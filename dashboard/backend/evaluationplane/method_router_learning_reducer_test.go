package evaluationplane

import (
	"fmt"
	"testing"
)

func routerLearningTestRecords() []executionRecordEvidence {
	records := make([]executionRecordEvidence, 0, 48)
	for _, policyID := range routerLearningPolicyIDs {
		for trial := 0; trial < routerLearningTrialCount; trial++ {
			trialID := fmt.Sprintf("trial-%02d", trial+1)
			for round := 0; round < 2; round++ {
				success := round == 0 || trial%2 == 1
				status := "failed"
				if success {
					status = "succeeded"
				}
				selected, selectionMethod := "arm-fast", policyID
				latency, cost := 10.0+float64(trial+round), 0.001+float64(round)*0.001
				records = append(records, executionRecordEvidence{
					TrackID: "joint", CaseID: []string{"round-01", "round-02"}[round],
					Status: status, SelectedArmID: &selected, SelectionMethod: &selectionMethod,
					Success: &success, LatencyMS: &latency, RuntimeCost: &cost,
					RouterLearning: &routerLearningMethodEvidence{
						ContractVersion: "evaluation-router-learning-method.v1", CorpusRevision: "router-learning-core-v1",
						PolicyID: policyID, TrialID: trialID, TrialSeed: int64(11 + trial), RoundIndex: int64(round),
						CandidateArmIDs: []string{"arm-fast", "arm-strong"}, EligibleArmIDs: []string{"arm-fast", "arm-strong"},
						ProposedArmID: selected, SelectedArmID: selected, OutcomeSuccess: success,
						FeedbackObserved: true, ProtectionRequired: round == 1, CallCount: 1,
						LifecycleCostUSD: cost, PropensityStatus: "unsupported",
					},
				})
			}
		}
	}
	return records
}

func TestRouterLearningReducerAttestsWorkerMetrics(t *testing.T) {
	learning, err := reduceRouterLearningMethod(routerLearningTestRecords())
	if err != nil {
		t.Fatal(err)
	}
	expected := methodMetricExpectations(
		methodRecordAttestation{RouterLearning: learning},
		[]TrackID{"joint"},
	)
	metrics := make([]Metric, 0, len(expected))
	for id, value := range expected {
		metrics = append(metrics, Metric{
			ID: id, Name: value.Name, TrackID: value.TrackID, Unit: value.Unit,
			Direction: value.Direction, Value: value.Value,
			ConfidenceInterval: value.Interval, SampleCount: value.SampleCount,
		})
	}
	report := Report{Run: Run{TrackIDs: []TrackID{"joint"}}, Metrics: metrics}
	if err := validateServerReducedMethodMetrics(report, methodRecordAttestation{RouterLearning: learning}); err != nil {
		t.Fatalf("valid Router Learning metrics were rejected: %v", err)
	}
	for index := range report.Metrics {
		if report.Metrics[index].ID == "joint.router_learning.static-base.solve_rate" {
			forged := *report.Metrics[index].Value + 0.1
			report.Metrics[index].Value = &forged
			break
		}
	}
	if err := validateServerReducedMethodMetrics(report, methodRecordAttestation{RouterLearning: learning}); err == nil {
		t.Fatal("forged Router Learning solve rate was accepted")
	}
}

func TestRouterLearningReducerRequiresPairedPolicyTrials(t *testing.T) {
	records := routerLearningTestRecords()
	records = records[:len(records)-1]
	if _, err := reduceRouterLearningMethod(records); err == nil {
		t.Fatal("unpaired Router Learning trials were accepted")
	}
}
