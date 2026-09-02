package config

import (
	"strings"
	"testing"
)

// TestImageGenPluginRejectedExplicitly guards the #3129 regression: a decision
// that still configures the removed image_gen route plugin must fail
// validation with an explicit unsupported-feature error, not silently pass.
func TestImageGenPluginRejectedExplicitly(t *testing.T) {
	err := validateDecisionPluginPayload(
		"d",
		0,
		DecisionPlugin{
			Type:          "image_gen",
			Configuration: MustStructuredPayload(map[string]interface{}{}),
		},
	)
	if err == nil {
		t.Fatal("expected image_gen plugin to be rejected")
	}
	if !strings.Contains(err.Error(), "image_gen route plugin was removed") {
		t.Fatalf("expected explicit removal error, got: %v", err)
	}
}
