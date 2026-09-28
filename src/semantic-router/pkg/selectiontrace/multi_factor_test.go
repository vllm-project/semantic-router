package selectiontrace

import (
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
