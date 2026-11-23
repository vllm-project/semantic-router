package selectiontrace

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultiFactorObjectiveClone(t *testing.T) {
	var absent *MultiFactorObjective
	if absent.Clone() != nil {
		t.Fatal("nil trace became recorded evidence")
	}
	zero, enabled := 0.0, true
	ref := config.ModelRef{
		Model: "model", LoRAName: "adapter", Weight: 2,
		ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &enabled, ReasoningEffort: "high"},
	}
	original := &MultiFactorObjective{
		Stages: []ObjectiveStage{{
			Factor: "latency", Metric: "ttft", Percentile: 95, Action: StageSkipped,
			Reason: IncompleteLatencyCoverage, Total: 2, Available: 1,
			Candidates: []CandidateValue{{Candidate: ref, Value: &zero}, {Candidate: config.ModelRef{Model: "unknown"}}},
		}},
		FinalSurvivors: []config.ModelRef{ref},
	}
	cloned := original.Clone()
	if !reflect.DeepEqual(original, cloned) {
		t.Fatal("clone changed evidence")
	}
	cloned.Stages[0].Action = StageApplied
	cloned.Stages[0].Candidates[0].Candidate.Model = "changed"
	*cloned.Stages[0].Candidates[0].Candidate.UseReasoning = false
	*cloned.Stages[0].Candidates[0].Value = 4
	cloned.Stages[0].Candidates[1].EliminationReason = MissingMeasurement
	cloned.FinalSurvivors[0].Model = "changed"
	*cloned.FinalSurvivors[0].UseReasoning = false
	if original.Stages[0].Action != StageSkipped || original.Stages[0].Candidates[0].Candidate.Model != "model" ||
		!enabled || zero != 0 || original.Stages[0].Candidates[1].EliminationReason != "" || original.FinalSurvivors[0].Model != "model" {
		t.Fatal("clone mutation crossed the persistence boundary")
	}
	for _, trace := range []*MultiFactorObjective{
		{},
		{Stages: []ObjectiveStage{}, FinalSurvivors: []config.ModelRef{}},
		{Stages: []ObjectiveStage{{}, {Candidates: []CandidateValue{}}}},
	} {
		if !reflect.DeepEqual(trace, trace.Clone()) {
			t.Fatal("clone changed nil/empty evidence shape")
		}
	}
}

func TestCandidateEstimatesClone(t *testing.T) {
	probability, tokens, cost, enabled := .86, 1180, .00041, true
	ref := config.ModelRef{Model: "qwen3-8b", ModelReasoningControl: config.ModelReasoningControl{UseReasoning: &enabled, ReasoningEffort: "high"}}
	original := &MultiFactorObjective{Estimates: &CandidateEstimates{Mode: "observe", Candidates: []CandidateEstimate{
		{Candidate: ref, Status: EstimateAvailable, SuccessProbability: &probability, ExpectedOutputTokens: &tokens, EstimatedCost: &cost, Currency: "USD", Source: EstimateSourceSystemOne, Deployment: "primary"},
		{Candidate: config.ModelRef{Model: "qwen3-8b"}, Status: EstimateDeadlineExceeded, Reason: "timeout", Source: EstimateSourceSystemOne},
	}}}
	cloned := original.Clone()
	if !reflect.DeepEqual(original, cloned) {
		t.Fatal("clone changed the estimates")
	}
	*cloned.Estimates.Candidates[0].SuccessProbability = 0
	*cloned.Estimates.Candidates[0].ExpectedOutputTokens = 0
	*cloned.Estimates.Candidates[0].EstimatedCost = 0
	*cloned.Estimates.Candidates[0].Candidate.UseReasoning = false
	cloned.Estimates.Candidates[1].Status = EstimateAvailable
	if probability != .86 || tokens != 1180 || cost != .00041 || !enabled || original.Estimates.Candidates[1].Status != EstimateDeadlineExceeded {
		t.Fatal("clone mutation crossed the persistence boundary")
	}
	for _, trace := range []*MultiFactorObjective{
		{Estimates: &CandidateEstimates{}},
		{Estimates: &CandidateEstimates{Candidates: []CandidateEstimate{}}},
	} {
		if !reflect.DeepEqual(trace, trace.Clone()) {
			t.Fatal("clone changed empty estimates")
		}
	}
}

// A weighted objective records no stages or survivors, so a trace of its
// estimates alone carries neither.
func TestEstimatesOnlyTraceOmitsObjectiveMembers(t *testing.T) {
	encoded, err := json.Marshal(&MultiFactorObjective{Estimates: &CandidateEstimates{Mode: "observe", Candidates: []CandidateEstimate{}}})
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"estimates":{"mode":"observe","candidates":[]}}` {
		t.Fatalf("encoded %s", encoded)
	}
}
