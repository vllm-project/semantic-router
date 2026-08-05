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
		ID: "rid",
		RouteDiagnostics: &RouteDiagnostics{
			Decision:             "agentic_session_route",
			PreviousModel:        "qwen/qwen3.6-rocm",
			SelectedModel:        "google/gemini-2.5-flash-lite",
			SessionPolicyApplied: true,
			SessionAction:        "switch",
		},
		Signals:         Signal{},
		ProjectionTrace: tr,
	}

	decoded := roundTripRecord(t, rec)
	assertProjectionTraceRoundTrip(t, decoded.ProjectionTrace, tr)
	assertRouteDiagnosticsRoundTrip(t, decoded.RouteDiagnostics)
}

func roundTripRecord(t *testing.T, rec Record) Record {
	t.Helper()

	data, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded Record
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return decoded
}

func assertProjectionTraceRoundTrip(
	t *testing.T,
	got *projectiontrace.Trace,
	want *projectiontrace.Trace,
) {
	t.Helper()

	if got == nil {
		t.Fatal("projection trace nil after round trip")
	}
	if got.SchemaVersion != want.SchemaVersion {
		t.Fatalf("schema version = %q", got.SchemaVersion)
	}
	if len(got.Partitions) != 1 || got.Partitions[0].Winner != "a" {
		t.Fatalf("partitions = %+v", got.Partitions)
	}
	if len(got.Scores) != 1 || got.Scores[0].Total != 1.5 {
		t.Fatalf("scores = %+v", got.Scores)
	}
	if len(got.Mappings) != 1 || got.Mappings[0].SelectedOutput != "out1" {
		t.Fatalf("mappings = %+v", got.Mappings)
	}
}

func assertRouteDiagnosticsRoundTrip(t *testing.T, got *RouteDiagnostics) {
	t.Helper()

	if got == nil || got.SessionAction != "switch" {
		t.Fatalf("route diagnostics = %+v", got)
	}
}

func TestCloneRecordCopiesProjectionTrace(t *testing.T) {
	tr := &projectiontrace.Trace{
		SchemaVersion: projectiontrace.SchemaVersion,
		Mappings: []projectiontrace.MappingDecision{
			{MappingName: "m", SourceScore: "s", ScoreValue: 0.5},
		},
	}
	rec := Record{
		ID: "1",
		RouteDiagnostics: &RouteDiagnostics{
			Decision:      "agentic_session_route",
			SelectedModel: "qwen/qwen3.6-rocm",
		},
		Signals:         Signal{},
		ProjectionTrace: tr,
	}
	cl := cloneRecord(rec)
	if cl.ProjectionTrace == nil || cl.ProjectionTrace == rec.ProjectionTrace {
		t.Fatal("expected deep-cloned projection trace pointer")
	}
	if cl.RouteDiagnostics == nil || cl.RouteDiagnostics == rec.RouteDiagnostics {
		t.Fatal("expected cloned route diagnostics pointer")
	}
	if len(cl.ProjectionTrace.Mappings) != 1 {
		t.Fatalf("mappings len = %d", len(cl.ProjectionTrace.Mappings))
	}
}
