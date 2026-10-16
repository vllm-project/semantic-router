package store

import (
	"context"
	"testing"
)

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

func TestMemoryStoreRetainsMatchedAgenticFactsSignals(t *testing.T) {
	memory := NewMemoryStore(10, 0)
	signals := Signal{AgenticFacts: []string{"reviewer_role", "verify_phase"}}
	id, err := memory.Add(context.Background(), Record{ID: "agentic-facts", Signals: signals})
	if err != nil {
		t.Fatal(err)
	}
	signals.AgenticFacts[0] = "changed"

	record, found, err := memory.Get(context.Background(), id)
	if err != nil || !found {
		t.Fatalf("get record: found=%v err=%v", found, err)
	}
	assertAgenticFactsSignals(t, "get", record.Signals.AgenticFacts)

	records, err := memory.List(context.Background())
	if err != nil || len(records) != 1 {
		t.Fatalf("list records: len=%d err=%v", len(records), err)
	}
	assertAgenticFactsSignals(t, "list", records[0].Signals.AgenticFacts)
}

func assertAgenticFactsSignals(t *testing.T, source string, got []string) {
	t.Helper()
	if len(got) != 2 || got[0] != "reviewer_role" || got[1] != "verify_phase" {
		t.Fatalf("%s agentic_facts signals = %v, want [reviewer_role verify_phase]", source, got)
	}
}
