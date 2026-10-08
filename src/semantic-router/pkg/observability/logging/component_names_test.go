package logging

import "testing"

func TestRenamedComponentNamesEveryLaterEvent(t *testing.T) {
	logs := observeGlobal(t)
	t.Cleanup(func() { renamedComponents.Store(nil) })

	ComponentEvent("extproc", "before", nil)
	RenameComponent("extproc", "router")
	ComponentEvent("extproc", "routing_decision", map[string]interface{}{"decision": "math"})
	ComponentWarnEvent("extproc", "warned", nil)
	ComponentEventFunc("extproc", "built_on_write", func() map[string]interface{} { return nil })
	ComponentEvent("gateway", "access", nil)
	ComponentEvent("extproc", "own_component", map[string]interface{}{"component": "explicit"})

	want := []string{"extproc", "router", "router", "router", "gateway", "explicit"}
	entries := logs.All()
	if len(entries) != len(want) {
		t.Fatalf("got %d entries, want %d", len(entries), len(want))
	}
	for i, entry := range entries {
		if got := entry.ContextMap()["component"]; got != want[i] {
			t.Errorf("event %q: component %v, want %q", entry.Message, got, want[i])
		}
	}
}
