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
			{MappingName: "m1", SourceScore: "s1", ScoreValue: 1.5, SelectedOutput: "out1", Confidence: 0.9},
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
	if decoded.ProjectionTrace == nil {
		t.Fatal("projection trace nil after round trip")
	}
	if decoded.ProjectionTrace.SchemaVersion != tr.SchemaVersion {
		t.Fatalf("schema version = %q", decoded.ProjectionTrace.SchemaVersion)
	}
	if len(decoded.ProjectionTrace.Partitions) != 1 || decoded.ProjectionTrace.Partitions[0].Winner != "a" {
		t.Fatalf("partitions = %+v", decoded.ProjectionTrace.Partitions)
	}
	if len(decoded.ProjectionTrace.Scores) != 1 || decoded.ProjectionTrace.Scores[0].Total != 1.5 {
		t.Fatalf("scores = %+v", decoded.ProjectionTrace.Scores)
	}
	if len(decoded.ProjectionTrace.Mappings) != 1 || decoded.ProjectionTrace.Mappings[0].SelectedOutput != "out1" {
		t.Fatalf("mappings = %+v", decoded.ProjectionTrace.Mappings)
	}
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
