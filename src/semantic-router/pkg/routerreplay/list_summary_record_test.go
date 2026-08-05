package routerreplay

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

func TestListSummaryRecordStripsLargeFields(t *testing.T) {
	huge := strings.Repeat("z", 1000)
	rec := RoutingRecord{
		ID:                "r1",
		RequestBody:       huge,
		ResponseBody:      huge,
		Prompt:            huge,
		ToolDefinitions:   huge,
		ProjectionTrace:   &projectiontrace.Trace{SchemaVersion: projectiontrace.SchemaVersion},
		ToolTrace:         &ToolTrace{Flow: "f", Steps: []ToolTraceStep{{ToolName: "t1", RawOutput: huge}}},
		Decision:          "d1",
		OriginalModel:     "m1",
		SelectedModel:     "m2",
		SignalConfidences: map[string]float64{"a": 0.5},
	}
	out := ListSummaryRecord(rec)
	if out.RequestBody != "" || out.ResponseBody != "" || out.Prompt != "" || out.ToolDefinitions != "" {
		t.Fatalf("expected large string fields cleared, got req=%d resp=%d", len(out.RequestBody), len(out.ResponseBody))
	}
	if out.ProjectionTrace != nil {
		t.Fatalf("expected projection trace cleared")
	}
	if out.ToolTrace == nil || len(out.ToolTrace.Steps) != 0 {
		t.Fatalf("expected tool trace steps stripped, got %#v", out.ToolTrace)
	}
	if len(out.ToolTrace.ToolNames) != 1 || out.ToolTrace.ToolNames[0] != "t1" {
		t.Fatalf("expected tool name preserved, got %#v", out.ToolTrace.ToolNames)
	}
	if out.Decision != "d1" || out.OriginalModel != "m1" {
		t.Fatalf("expected metadata preserved")
	}
}
