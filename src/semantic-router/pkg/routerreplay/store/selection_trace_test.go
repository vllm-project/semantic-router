package store

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

func TestSelectionTracePersistsKnownZeroAndUnknownSeparately(t *testing.T) {
	zero := 0.0
	trace := &selectiontrace.MultiFactorObjective{
		FinalSurvivors: []config.ModelRef{{Model: "cheap"}},
		Stages: []selectiontrace.ObjectiveStage{{
			Factor: "latency", Metric: "ttft", Action: selectiontrace.StageSkipped,
			Reason: selectiontrace.IncompleteLatencyCoverage, Available: 1, Total: 2,
			Candidates: []selectiontrace.CandidateValue{
				{Candidate: config.ModelRef{Model: "known"}, Value: &zero},
				{Candidate: config.ModelRef{Model: "cheap"}},
			},
		}},
	}
	record := Record{ID: "selection-evidence", RouteDiagnostics: &RouteDiagnostics{SelectionTrace: trace}}
	decoded := roundTripRecord(t, record)
	if !reflect.DeepEqual(decoded.RouteDiagnostics.SelectionTrace, trace) {
		t.Fatalf("selection evidence changed after storage: %+v", decoded.RouteDiagnostics.SelectionTrace)
	}
	cloned := cloneRecord(record)
	*cloned.RouteDiagnostics.SelectionTrace.Stages[0].Candidates[0].Value = 99
	cloned.RouteDiagnostics.SelectionTrace.FinalSurvivors[0].Model = "changed"
	if *trace.Stages[0].Candidates[0].Value != 0 || trace.FinalSurvivors[0].Model != "cheap" {
		t.Fatal("cloned replay mutated original selection evidence")
	}
}
