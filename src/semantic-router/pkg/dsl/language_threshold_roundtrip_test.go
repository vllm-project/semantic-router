package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecompileLanguageThresholdRoundTrip(t *testing.T) {
	cfg := &config.RouterConfig{
		Categories: []config.Category{},
		LanguageRules: []config.LanguageRule{
			{Name: "es", Description: "Spanish", Threshold: 0.8},
		},
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting failed: %v", err)
	}

	if !strings.Contains(dslText, "SIGNAL language es") {
		t.Fatalf("decompiled DSL missing language signal:\n%s", dslText)
	}
	if !strings.Contains(dslText, "threshold: 0.8") {
		t.Fatalf("decompiled DSL missing threshold field:\n%s", dslText)
	}

	// Round-trip: compile back and verify threshold survives
	rtCfg, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("round-trip compile errors: %v\n%s", errs, dslText)
	}
	if len(rtCfg.LanguageRules) != 1 {
		t.Fatalf("round-trip: expected 1 language rule, got %d", len(rtCfg.LanguageRules))
	}
	if rtCfg.LanguageRules[0].Threshold != 0.8 {
		t.Fatalf("round-trip: expected threshold 0.8, got %v", rtCfg.LanguageRules[0].Threshold)
	}
}
