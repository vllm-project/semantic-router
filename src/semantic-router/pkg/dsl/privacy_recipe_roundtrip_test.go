package dsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedPrivacyRecipeParsesAndDecompilesWithoutError(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.yaml")
	cfg, err := config.Parse(assetPath)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	if !strings.Contains(dslText, "SIGNAL taxonomy") {
		t.Error("decompiled DSL missing taxonomy signal")
	}
	if !strings.Contains(dslText, "TOOL_SCOPE") {
		t.Error("decompiled DSL missing TOOL_SCOPE directives")
	}
}

func TestMaintainedPrivacyRecipeDSLRoundTrip(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("privacy recipe DSL not found at %s: %v", assetPath, err)
	}

	input := string(data)

	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.TaxonomyRules) == 0 {
		t.Error("expected at least one taxonomy rule from privacy recipe")
	}

	hasToolScope := false
	for _, dec := range cfg.Decisions {
		if dec.ToolScope != "" {
			hasToolScope = true
			break
		}
	}
	if !hasToolScope {
		t.Error("expected at least one decision with ToolScope from privacy recipe")
	}

	// Decompile back to DSL
	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	// Re-compile to verify round-trip
	cfg2, errs2 := Compile(dslText)
	if len(errs2) > 0 {
		t.Fatalf("Round-trip compile errors: %v", errs2)
	}

	if len(cfg2.TaxonomyRules) != len(cfg.TaxonomyRules) {
		t.Errorf("round-trip taxonomy rules: %d → %d",
			len(cfg.TaxonomyRules), len(cfg2.TaxonomyRules))
	}
	if len(cfg2.Decisions) != len(cfg.Decisions) {
		t.Errorf("round-trip Decisions: %d → %d",
			len(cfg.Decisions), len(cfg2.Decisions))
	}

	for i, dec := range cfg.Decisions {
		if i < len(cfg2.Decisions) && dec.ToolScope != cfg2.Decisions[i].ToolScope {
			t.Errorf("round-trip decision %q ToolScope: %q → %q",
				dec.Name, dec.ToolScope, cfg2.Decisions[i].ToolScope)
		}
	}
}

func TestMaintainedPrivacyRecipeValidates(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "privacy", "privacy-router.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("privacy recipe DSL not found: %v", err)
	}

	diags, errs := Validate(string(data))
	if len(errs) > 0 {
		t.Fatalf("Validate parse errors: %v", errs)
	}

	for _, diag := range diags {
		if diag.Level == DiagError {
			t.Errorf("validation error: %s", diag.Message)
		}
	}
}
