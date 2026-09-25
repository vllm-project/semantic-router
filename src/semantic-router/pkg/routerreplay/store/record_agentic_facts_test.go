package store

import "testing"

func TestCloneRecordClonesAgenticFactsReasons(t *testing.T) {
	record := Record{
		RouteDiagnostics: &RouteDiagnostics{
			AgenticFactsStatus:  "rejected",
			AgenticFactsReasons: []string{"expires_at:expired"},
		},
	}
	cloned := cloneRecord(record)
	cloned.RouteDiagnostics.AgenticFactsStatus = "accepted"
	cloned.RouteDiagnostics.AgenticFactsReasons[0] = "changed"

	if record.RouteDiagnostics.AgenticFactsStatus != "rejected" {
		t.Fatalf("original agentic facts status mutated: %q", record.RouteDiagnostics.AgenticFactsStatus)
	}
	if record.RouteDiagnostics.AgenticFactsReasons[0] != "expires_at:expired" {
		t.Fatalf("original agentic facts reasons mutated: %v", record.RouteDiagnostics.AgenticFactsReasons)
	}
}
