package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
						ContractVersion: "evaluation-router-learning-method.v1", CorpusRevision: "router-learning-core-v2",
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

func TestRouterLearningRejectsSimplifiedPolicyLabel(t *testing.T) {
	record := routerLearningTestRecords()[0]
	method := *record.RouterLearning
	method.PolicyID = "routing-sampling"
	record.SelectionMethod = &method.PolicyID
	if err := validateRouterLearningMethod(method, record); err != nil {
		t.Fatalf("production policy was rejected: %v", err)
	}
	method.PolicyID = "simplified-routing-sampling"
	if err := validateRouterLearningMethod(method, record); err == nil {
		t.Fatal("simplified policy label was accepted for production replay evidence")
	}
}

func TestRouterLearningPythonEvidenceParity(t *testing.T) {
	root := os.Getenv("VLLM_SR_ROUTER_LEARNING_PARITY_BUNDLE")
	if root == "" {
		t.Skip("run test_router_learning_benchmark.py for Python reducer parity")
	}
	data, err := os.ReadFile(filepath.Join(root, "parity-records.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var records []executionRecordEvidence
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		var record executionRecordEvidence
		if decodeErr := json.Unmarshal(line, &record); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		if record.RouterLearning == nil {
			t.Fatal("missing Python method evidence")
		}
		if validationErr := validateRouterLearningMethod(*record.RouterLearning, record); validationErr != nil {
			t.Fatal(validationErr)
		}
		records = append(records, record)
	}
	data, err = os.ReadFile(filepath.Join(root, "parity-plan.json"))
	if err != nil {
		t.Fatal(err)
	}
	var plan struct {
		Seed    int64
		CaseIDs []string
	}
	if planErr := json.Unmarshal(data, &plan); planErr != nil {
		t.Fatal(planErr)
	}
	planned := make(map[string]struct{})
	for _, id := range plan.CaseIDs {
		planned[id] = struct{}{}
	}
	if planErr := validateRouterLearningRunPlan(records, planned, plan.Seed); planErr != nil {
		t.Fatal(planErr)
	}
	learning, reduceErr := reduceRouterLearningMethod(records)
	if reduceErr != nil {
		t.Fatal(reduceErr)
	}
	data, err = os.ReadFile(filepath.Join(root, "parity-metrics.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metrics []Metric
	if decodeErr := json.Unmarshal(data, &metrics); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	report := Report{Run: Run{TrackIDs: []TrackID{"joint"}}, Metrics: metrics}
	if attestErr := validateServerReducedMethodMetrics(report, methodRecordAttestation{RouterLearning: learning}); attestErr != nil {
		t.Fatal(attestErr)
	}
}
