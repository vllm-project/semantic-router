package config

import (
	"strings"
	"testing"
)

func TestParseRoutingYAMLBytesRequiresExactDocumentWrapper(t *testing.T) {
	for _, source := range []string{
		"version: v0.4\nmodels: []\nrecipes: []\nentrypoints: []\n",
		"document: {}\nversion: v0.4\n",
		"decisions: []\n",
	} {
		if _, err := ParseRoutingYAMLBytes([]byte(source)); err == nil ||
			!strings.Contains(err.Error(), "exactly one top-level document field") {
			t.Fatalf("ParseRoutingYAMLBytes(%q) error = %v", source, err)
		}
	}
}

func TestParseRoutingYAMLBytesRejectsUnknownDocumentField(t *testing.T) {
	_, err := ParseRoutingYAMLBytes([]byte("document:\n  decisions: []\n  technical_explanation: leaked\n"))
	if err == nil || !strings.Contains(err.Error(), "field technical_explanation not found") {
		t.Fatalf("ParseRoutingYAMLBytes() error = %v", err)
	}
}

func TestParseRoutingYAMLBytesPreservesRecipeStrategy(t *testing.T) {
	cfg, err := ParseRoutingYAMLBytes([]byte(`
document:
  strategy: confidence
  decisions:
    - name: direct
      rules: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Strategy != RoutingStrategyConfidence {
		t.Fatalf("strategy = %q, want %q", cfg.Strategy, RoutingStrategyConfidence)
	}
	roundTrip := CanonicalRoutingFromRouterConfig(cfg)
	if roundTrip.Strategy != RoutingStrategyConfidence {
		t.Fatalf("round-trip strategy = %q, want %q", roundTrip.Strategy, RoutingStrategyConfidence)
	}
}
