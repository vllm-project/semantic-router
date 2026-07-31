package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestComplexityMethodRoundTrip verifies that a complexity rule's method survives
// decompile -> compile so DSL authoring of method: model is lossless.
func TestComplexityMethodRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ComplexityRules: []config.ComplexityRule{
					{
						Name:        "needs_reasoning_model",
						Method:      config.ComplexityMethodModel,
						Threshold:   0.6,
						Description: "Trained classifier complexity rule.",
					},
				},
			},
		},
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}
	if !strings.Contains(dslText, `method: "model"`) {
		t.Fatalf("decompiled DSL missing method: \"model\":\n%s", dslText)
	}

	roundTripped, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if len(roundTripped.ComplexityRules) != 1 {
		t.Fatalf("expected 1 complexity rule, got %d", len(roundTripped.ComplexityRules))
	}
	if got := roundTripped.ComplexityRules[0].Method; got != config.ComplexityMethodModel {
		t.Fatalf("round-trip complexity method = %q, want %q", got, config.ComplexityMethodModel)
	}
}
