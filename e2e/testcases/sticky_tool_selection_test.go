package testcases

import (
	"encoding/json"
	"testing"
)

func TestAssertStickyInvalidationAllowsStatelessToolChoice(t *testing.T) {
	previous := stickyTestSnapshot(
		`{"type":"function","function":{"name":"search_web","description":"revision 1"}}`,
		`{"type":"function","function":{"name":"calculate"}}`,
	)

	tests := []struct {
		name     string
		updated  stickyToolSnapshot
		baseline stickyToolSnapshot
	}{
		{
			name:     "changed tool remains relevant",
			updated:  stickyTestSnapshot(`{"type":"function","function":{"name":"search_web","description":"revision 2"}}`),
			baseline: stickyTestSnapshot(`{"type":"function","function":{"name":"search_web","description":"revision 2"}}`),
		},
		{
			name:     "different offered tool is more relevant",
			updated:  stickyTestSnapshot(`{"type":"function","function":{"name":"get_weather"}}`),
			baseline: stickyTestSnapshot(`{"type":"function","function":{"name":"get_weather"}}`),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := assertStickyInvalidation(previous, test.updated, test.baseline, "search_web"); err != nil {
				t.Fatalf("assertStickyInvalidation() error = %v", err)
			}
		})
	}
}

func TestAssertStickyInvalidationRejectsStaleDefinition(t *testing.T) {
	stale := `{"type":"function","function":{"name":"search_web","description":"revision 1"}}`
	previous := stickyTestSnapshot(
		stale,
		`{"type":"function","function":{"name":"calculate"}}`,
	)
	updated := stickyTestSnapshot(stale)

	if err := assertStickyInvalidation(previous, updated, updated, "search_web"); err == nil {
		t.Fatal("assertStickyInvalidation() error = nil, want stale definition rejection")
	}
}

func TestAssertStickyInvalidationRejectsBaselineMismatch(t *testing.T) {
	previous := stickyTestSnapshot(`{"type":"function","function":{"name":"search_web","description":"revision 1"}}`)
	updated := stickyTestSnapshot(`{"type":"function","function":{"name":"search_web","description":"revision 2"}}`)
	baseline := stickyTestSnapshot(`{"type":"function","function":{"name":"get_weather"}}`)

	if err := assertStickyInvalidation(previous, updated, baseline, "search_web"); err == nil {
		t.Fatal("assertStickyInvalidation() error = nil, want baseline mismatch rejection")
	}
}

func stickyTestSnapshot(definitions ...string) stickyToolSnapshot {
	snapshot := stickyToolSnapshot{
		Tools: make([]json.RawMessage, len(definitions)),
		Names: make([]string, len(definitions)),
	}
	for i, definition := range definitions {
		snapshot.Tools[i] = json.RawMessage(definition)
		name, err := stickyProviderToolName(snapshot.Tools[i])
		if err != nil {
			panic(err)
		}
		snapshot.Names[i] = name
	}
	return snapshot
}
