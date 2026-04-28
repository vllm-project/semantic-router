package routerreplay

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func makeSteps(n int) []ToolTraceStep {
	steps := make([]ToolTraceStep, n)
	for i := range steps {
		steps[i] = ToolTraceStep{Type: "assistant_tool_call", ToolName: "tool"}
	}
	return steps
}

// TestSetMaxToolTraceStepsCapsBelowLimit verifies that records with fewer steps
// than the cap are stored unchanged.
func TestSetMaxToolTraceStepsCapsBelowLimit(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetMaxToolTraceSteps(50)

	_, err := recorder.AddRecord(RoutingRecord{
		ID:        "r1",
		Decision:  "d",
		RequestID: "req",
		ToolTrace: &ToolTrace{Steps: makeSteps(10)},
	})
	if err != nil {
		t.Fatalf("AddRecord failed: %v", err)
	}
}

// TestSetMaxToolTraceStepsCapsAboveLimit is the primary regression for #1835:
// a session with 200 tool-call steps must be truncated to the configured cap.
func TestSetMaxToolTraceStepsCapsAboveLimit(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetMaxToolTraceSteps(50)

	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "r2",
		Decision:  "d",
		RequestID: "req",
		ToolTrace: &ToolTrace{Steps: makeSteps(200)},
	})
	if err != nil {
		t.Fatalf("AddRecord failed: %v", err)
	}

	records := recorder.ListAllRecords()
	var found *RoutingRecord
	for i := range records {
		if records[i].ID == id {
			found = &records[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("record %q not found after AddRecord", id)
	}
	if found.ToolTrace == nil {
		t.Fatal("expected ToolTrace to be set")
	}
	if len(found.ToolTrace.Steps) != 50 {
		t.Errorf("expected steps capped at 50, got %d", len(found.ToolTrace.Steps))
	}
}

// TestSetMaxToolTraceStepsZeroMeansNoLimit confirms that 0 disables the cap.
func TestSetMaxToolTraceStepsZeroMeansNoLimit(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetMaxToolTraceSteps(0)

	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "r3",
		Decision:  "d",
		RequestID: "req",
		ToolTrace: &ToolTrace{Steps: makeSteps(500)},
	})
	if err != nil {
		t.Fatalf("AddRecord failed: %v", err)
	}

	records := recorder.ListAllRecords()
	var found *RoutingRecord
	for i := range records {
		if records[i].ID == id {
			found = &records[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("record %q not found", id)
	}
	if len(found.ToolTrace.Steps) != 500 {
		t.Errorf("expected all 500 steps with no cap, got %d", len(found.ToolTrace.Steps))
	}
}

// TestUpdateToolTraceCapsSteps ensures the cap is enforced on the update path
// (used when response-side traces are attached after request recording).
func TestUpdateToolTraceCapsSteps(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetMaxToolTraceSteps(30)

	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "r4",
		Decision:  "d",
		RequestID: "req",
	})
	if err != nil {
		t.Fatalf("AddRecord failed: %v", err)
	}

	err = recorder.UpdateToolTrace(id, ToolTrace{Steps: makeSteps(100)})
	if err != nil {
		t.Fatalf("UpdateToolTrace failed: %v", err)
	}

	records := recorder.ListAllRecords()
	var found *RoutingRecord
	for i := range records {
		if records[i].ID == id {
			found = &records[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("record %q not found", id)
	}
	if found.ToolTrace == nil {
		t.Fatal("expected ToolTrace to be set after update")
	}
	if len(found.ToolTrace.Steps) != 30 {
		t.Errorf("expected steps capped at 30 after update, got %d", len(found.ToolTrace.Steps))
	}
}

// TestCapToolTraceStepsNilTrace confirms nil ToolTrace is handled safely.
func TestCapToolTraceStepsNilTrace(t *testing.T) {
	rec := &RoutingRecord{ID: "r5", ToolTrace: nil}
	capToolTraceSteps(rec, 10) // must not panic
}
