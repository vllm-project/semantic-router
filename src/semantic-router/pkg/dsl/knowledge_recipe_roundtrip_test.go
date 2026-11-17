package dsl

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMaintainedKnowledgeRecipeDSLRoundTrip(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "knowledge", "recipe.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("mmlu recipe DSL not found at %s: %v", assetPath, err)
	}

	cfg, errs := Compile(string(data))
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.SystemOneRules) == 0 {
		t.Error("expected the subject question from the knowledge recipe")
	}
	if len(cfg.Decisions) == 0 {
		t.Error("expected at least one decision from mmlu recipe")
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	cfg2, errs2 := Compile(dslText)
	if len(errs2) > 0 {
		t.Fatalf("Round-trip compile errors: %v", errs2)
	}

	if len(cfg2.SystemOneRules) != len(cfg.SystemOneRules) {
		t.Errorf("round-trip decision questions: %d -> %d", len(cfg.SystemOneRules), len(cfg2.SystemOneRules))
	}
	if len(cfg2.Decisions) != len(cfg.Decisions) {
		t.Errorf("round-trip decisions: %d -> %d", len(cfg.Decisions), len(cfg2.Decisions))
	}
}

func TestMaintainedKnowledgeRecipeValidates(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "knowledge", "recipe.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("mmlu recipe DSL not found: %v", err)
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
