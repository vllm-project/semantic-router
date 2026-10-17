package routerreplay

import (
	"reflect"
	"testing"
	"time"
)

// The receipt carries only an outcome, a reason, and counts; its Replay form
// adds nothing beyond them.
func TestStickyToolSelectionReceiptIsContentFree(t *testing.T) {
	fields := reflect.TypeOf(StickyToolSelectionReceipt{})
	want := []string{"Outcome", "Reason", "Selected", "Reused", "Added", "Pinned", "Removed"}
	if fields.NumField() != len(want) {
		t.Fatalf("receipt has %d fields, want exactly %v", fields.NumField(), want)
	}
	for i, name := range want {
		if fields.Field(i).Name != name {
			t.Fatalf("field %d = %s, want %s", i, fields.Field(i).Name, name)
		}
	}
	now := time.Date(2026, 10, 9, 12, 0, 0, 0, time.FixedZone("x", 3600))
	outcome := StickyToolSelectionReceipt{Outcome: "updated", Reason: "", Selected: 3, Reused: 2, Added: 1, Pinned: 1, Removed: 0}.ReplayOutcome(now)
	if outcome.Target != StickyToolSelectionTarget || outcome.Verdict != "updated" || !outcome.Timestamp.Equal(now) || outcome.Timestamp.Location() != time.UTC {
		t.Fatalf("unexpected outcome %+v", outcome)
	}
	wantMetadata := map[string]string{"selected": "3", "reused": "2", "added": "1", "pinned": "1", "removed": "0"}
	if !reflect.DeepEqual(outcome.Metadata, wantMetadata) {
		t.Fatalf("metadata = %v, want %v", outcome.Metadata, wantMetadata)
	}
	if outcome.TargetRef != "" || outcome.IdempotencyKey != "" || outcome.Score != 0 {
		t.Fatalf("receipt outcome must not carry references or scores: %+v", outcome)
	}
}
