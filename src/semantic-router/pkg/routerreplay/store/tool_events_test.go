package store

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func fileLines(t *testing.T, path string) int {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Count(data, []byte{'\n'})
}

func toolEvent(session, call string, received time.Time) ToolEvent {
	return ToolEvent{SessionID: session, ToolCallID: call, Tool: "Bash", Status: ToolEventSucceeded, ReceivedAt: received}
}

// A later report for the same session and call fills in what the first one
// left out, sessions never mix, and the store keeps at most its limit of
// events and none older than its TTL.
func TestToolEventStoreMergesPerCallAndStaysBounded(t *testing.T) {
	now := time.Now()
	started, asked := now.Add(-3*time.Second), now.Add(-4*time.Second)
	store := NewToolEventStore(3, time.Hour)
	first := toolEvent("s1", "call-1", now)
	first.ApprovalRequestedAt = &asked
	later := ToolEvent{SessionID: "s1", ToolCallID: "call-1", StartedAt: &started, ReceivedAt: now}
	if err := store.Add([]ToolEvent{first, toolEvent("s2", "call-1", now), later}); err != nil {
		t.Fatal(err)
	}
	got := store.Session("s1")["call-1"]
	if got.Tool != "Bash" || got.Status != ToolEventSucceeded || got.StartedAt == nil || got.ApprovalRequestedAt == nil {
		t.Fatalf("merged event = %+v", got)
	}
	if len(store.Session("s1")) != 1 || len(store.Session("s2")) != 1 || store.Session("") != nil {
		t.Fatal("sessions mixed")
	}
	_ = store.Add([]ToolEvent{toolEvent("s3", "a", now), toolEvent("s3", "b", now)})
	// s1 was reported last, so s2 is the oldest.
	if len(store.Session("s2")) != 0 || len(store.Session("s1")) != 1 || len(store.Session("s3")) != 2 {
		t.Fatal("the oldest event was not evicted at the limit")
	}
	// The Router stamps each event as it takes it, so the oldest come first.
	aging := NewToolEventStore(10, time.Hour)
	_ = aging.Add([]ToolEvent{toolEvent("s4", "old", now.Add(-2*time.Hour)), toolEvent("s4", "new", now)})
	if kept := aging.Session("s4"); len(kept) != 1 || kept["new"].ToolCallID != "new" {
		t.Fatal("an event past the TTL is still kept")
	}
}

// The file keeps the events across a reopen, without those past the TTL, in a
// file only its owner reads; it is rewritten once it holds twice the limit, a
// line cut short is skipped, and every opener in the process shares one store.
func TestToolEventFileKeepsEventsAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".vllm-sr", "router-tool-events.jsonl")
	open := func() *ToolEventStore {
		t.Helper()
		store, err := OpenToolEventStore(path, 4, time.Hour)
		if err != nil {
			t.Fatal(err)
		}
		return store
	}
	now := time.Now()
	store := open()
	if shared := open(); shared != store {
		t.Fatal("a second opener got a store of its own")
	}
	_ = store.Add([]ToolEvent{toolEvent("s", "expired", now.Add(-2*time.Hour)), toolEvent("s", "kept", now)})
	if err := store.Close(); err != nil || store.disk == nil {
		t.Fatalf("closing one share closed the store: %v", err)
	}
	for i := range 8 {
		_ = store.Add([]ToolEvent{toolEvent("s", string(rune('a'+i)), now)})
	}
	if lines := fileLines(t, path); lines > 8 {
		t.Fatalf("file holds %d lines, want a rewrite at twice the limit", lines)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("file mode = %v, %v", info, err)
	}
	file, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	_, _ = file.WriteString(`{"session_id":"s","tool_call_id":"cut`)
	_ = file.Close()
	reopened := open()
	defer func() { _ = reopened.Close() }()
	kept := reopened.Session("s")
	if len(kept) != 4 || kept["h"].Tool != "Bash" {
		t.Fatalf("reopened store holds %v", kept)
	}
	if _, ok := kept["expired"]; ok {
		t.Fatal("an event past the TTL came back")
	}
}
