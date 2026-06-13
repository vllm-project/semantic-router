package store

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

func TestRecordJSONRoundTripProjectionTrace(t *testing.T) {
	tr := &projectiontrace.Trace{
		SchemaVersion: projectiontrace.SchemaVersion,
		Partitions: []projectiontrace.PartitionResolution{
			{
				GroupName: "g1", SignalType: "domain", Winner: "a", Margin: 0.12, WinnerScore: 0.88,
			},
		},
		Scores: []projectiontrace.ScoreBreakdown{
			{Name: "s1", Total: 1.5, Inputs: []projectiontrace.ScoreInputPart{
				{Type: "keyword", Name: "k", Weight: 1, Value: 1.5, Contribution: 1.5},
			}},
		},
		Mappings: []projectiontrace.MappingDecision{
			{
				MappingName:    "m1",
				SourceScore:    "s1",
				ScoreValue:     1.5,
				MatchedOutputs: []string{"out1", "out2"},
				SelectedOutput: "out1",
				Confidence:     0.9,
			},
		},
	}
	rec := Record{
		ID:              "rid",
		Signals:         Signal{},
		ProjectionTrace: tr,
	}

	data, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded Record
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	assertProjectionTraceRoundTrip(t, decoded.ProjectionTrace, tr.SchemaVersion)
}

func TestCloneRecordCopiesProjectionTrace(t *testing.T) {
	tr := &projectiontrace.Trace{
		SchemaVersion: projectiontrace.SchemaVersion,
		Mappings: []projectiontrace.MappingDecision{
			{MappingName: "m", SourceScore: "s", ScoreValue: 0.5},
		},
	}
	rec := Record{ID: "1", Signals: Signal{}, ProjectionTrace: tr}
	cl := cloneRecord(rec)
	if cl.ProjectionTrace == nil || cl.ProjectionTrace == rec.ProjectionTrace {
		t.Fatal("expected deep-cloned projection trace pointer")
	}
	if len(cl.ProjectionTrace.Mappings) != 1 {
		t.Fatalf("mappings len = %d", len(cl.ProjectionTrace.Mappings))
	}
}

func assertProjectionTraceRoundTrip(t *testing.T, got *projectiontrace.Trace, wantSchema string) {
	t.Helper()

	if got == nil {
		t.Fatal("projection trace nil after round trip")
	}
	if got.SchemaVersion != wantSchema {
		t.Fatalf("schema version = %q", got.SchemaVersion)
	}

	if len(got.Partitions) != 1 {
		t.Fatalf("partitions = %+v", got.Partitions)
	}
	if got.Partitions[0].Winner != "a" {
		t.Fatalf("partitions = %+v", got.Partitions)
	}

	if len(got.Scores) != 1 {
		t.Fatalf("scores = %+v", got.Scores)
	}
	if got.Scores[0].Total != 1.5 {
		t.Fatalf("scores = %+v", got.Scores)
	}

	if len(got.Mappings) != 1 {
		t.Fatalf("mappings = %+v", got.Mappings)
	}
	if got.Mappings[0].SelectedOutput != "out1" {
		t.Fatalf("mappings = %+v", got.Mappings)
	}
	if len(got.Mappings[0].MatchedOutputs) != 2 {
		t.Fatalf("matched outputs = %+v", got.Mappings[0].MatchedOutputs)
	}
	if got.Mappings[0].MatchedOutputs[1] != "out2" {
		t.Fatalf("matched outputs = %+v", got.Mappings[0].MatchedOutputs)
	}
}
