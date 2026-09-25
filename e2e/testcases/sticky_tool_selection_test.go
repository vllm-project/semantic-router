package testcases

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestStickyProviderPrefixUsesStableAuthorizedCatalog(t *testing.T) {
	first := stickyProviderPrefixTools()
	second := stickyProviderPrefixTools()
	if len(first) != 2 {
		t.Fatalf("provider-prefix catalog length = %d, want 2", len(first))
	}
	if first[0].Name != "calculate" || first[1].Name != "get_weather" {
		t.Fatalf("provider-prefix catalog = %v, want [calculate get_weather]", []string{first[0].Name, first[1].Name})
	}
	if first[1].CacheControl["type"] != "ephemeral" {
		t.Fatalf("weather cache control = %v, want ephemeral", first[1].CacheControl)
	}
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal first catalog: %v", err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatalf("marshal second catalog: %v", err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatalf("provider-prefix catalogs differ:\nfirst:  %s\nsecond: %s", firstJSON, secondJSON)
	}
	if strings.TrimSpace(stickyProviderPrefixFirstPrompt) == "" {
		t.Fatal("provider-prefix first prompt must select one relevant tool")
	}
	if strings.TrimSpace(stickyProviderPrefixGrowthPrompt) != "" {
		t.Fatal("provider-prefix growth prompt must pass through the full authorized catalog")
	}
}

func TestAssertStickyTrustedProviderPrefixReportsStructureBeforeUsage(t *testing.T) {
	cycle := stickyPrefixCycle{
		FirstTools: stickyTestSnapshot(`{"name":"get_weather"}`),
		SecondTools: stickyTestSnapshot(
			`{"name":"calculate"}`,
			`{"name":"get_weather"}`,
		),
	}

	err := assertStickyTrustedProviderPrefix(cycle)
	if err == nil || !strings.Contains(err.Error(), "trusted growth reordered") {
		t.Fatalf("assertStickyTrustedProviderPrefix() error = %v, want tool-order failure", err)
	}
}

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
