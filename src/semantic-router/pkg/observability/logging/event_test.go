package logging

import (
	"testing"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

// observeGlobal routes the global logger to an observer behind a sampler that
// keeps the first two entries of each message, as the production sampler
// keeps the first of each second.
func observeGlobal(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.InfoLevel)
	restore := zap.ReplaceGlobals(zap.New(zapcore.NewSamplerWithOptions(core, time.Hour, 2, 0)))
	t.Cleanup(restore)
	return logs
}

func TestComponentEventCarriesEventAndComponent(t *testing.T) {
	logs := observeGlobal(t)
	ComponentEvent("gateway", "access", map[string]interface{}{"response_code": 200, "path": "/v1/models"})
	ComponentEvent("gateway", "custom", map[string]interface{}{"event": "own", "component": "own-component"})
	LogEvent("plain", map[string]interface{}{"n": int64(3)})

	entries := logs.All()
	if len(entries) != 3 {
		t.Fatalf("got %d entries, want 3", len(entries))
	}
	want := []map[string]interface{}{
		{"response_code": int64(200), "path": "/v1/models", "event": "access", "component": "gateway"},
		{"event": "own", "component": "own-component"},
		{"n": int64(3), "event": "plain"},
	}
	for i, entry := range entries {
		got := entry.ContextMap()
		if len(got) != len(want[i]) {
			t.Errorf("entry %d fields = %v, want %v", i, got, want[i])
			continue
		}
		for key, value := range want[i] {
			if got[key] != value {
				t.Errorf("entry %d field %s = %v, want %v", i, key, got[key], value)
			}
		}
	}
	if entries[0].Message != "access" || entries[0].Level != zapcore.InfoLevel {
		t.Errorf("entry 0 = %s %q, want info \"access\"", entries[0].Level, entries[0].Message)
	}
}

// Sampling decides before the fields are built, with the same counts as
// before: the first two of a message pass, the rest are dropped.
func TestEventsKeepTheSamplerDecision(t *testing.T) {
	logs := observeGlobal(t)
	for i := 0; i < 5; i++ {
		ComponentEvent("gateway", "access", map[string]interface{}{"i": i})
		WarnEvent("warned", nil)
	}
	DebugEvent("disabled", map[string]interface{}{"i": 1})
	if got := logs.FilterMessage("access").Len(); got != 2 {
		t.Errorf("access entries = %d, want 2", got)
	}
	if got := logs.FilterMessage("warned").Len(); got != 2 {
		t.Errorf("warned entries = %d, want 2", got)
	}
	if got := logs.FilterMessage("disabled").Len(); got != 0 {
		t.Errorf("debug entries = %d, want 0", got)
	}
}

// An event the sampler drops costs no encoding: only the caller's own fields
// are allocated.
func TestDroppedEventsDoNotAllocate(t *testing.T) {
	observeGlobal(t)
	fields := map[string]interface{}{"path": "/v1/chat/completions", "response_code": 200}
	ComponentEvent("gateway", "access", fields)
	ComponentEvent("gateway", "access", fields)
	if allocs := testing.AllocsPerRun(100, func() { ComponentEvent("gateway", "access", fields) }); allocs != 0 {
		t.Errorf("a dropped event allocates %v times, want 0", allocs)
	}
}

// ComponentEventFunc writes what ComponentEvent writes, and builds the fields
// only for the entries the sampler keeps.
func TestComponentEventFuncBuildsOnlyWrittenEntries(t *testing.T) {
	logs := observeGlobal(t)
	built := 0
	for i := 0; i < 5; i++ {
		ComponentEventFunc("gateway", "access", func() map[string]interface{} {
			built++
			return map[string]interface{}{"response_code": 200}
		})
	}
	if built != 2 {
		t.Errorf("fields built %d times, want 2", built)
	}
	entries := logs.FilterMessage("access").All()
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
	got := entries[0].ContextMap()
	want := map[string]interface{}{"response_code": int64(200), "event": "access", "component": "gateway"}
	if len(got) != len(want) {
		t.Fatalf("fields = %v, want %v", got, want)
	}
	for key, value := range want {
		if got[key] != value {
			t.Errorf("field %s = %v, want %v", key, got[key], value)
		}
	}
}
