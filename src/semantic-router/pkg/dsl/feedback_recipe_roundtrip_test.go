package dsl

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedFeedbackRecipeParsesAndDecompilesWithoutError(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "feedback", "feedback-router.yaml")
	cfg, err := config.Parse(assetPath)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	if dslText == "" {
		t.Fatal("decompiled DSL should not be empty")
	}
}

func TestMaintainedFeedbackRecipeDSLRoundTrip(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "feedback", "feedback-router.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("feedback recipe DSL not found at %s: %v", assetPath, err)
	}

	cfg, errs := Compile(string(data))
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.ReaskRules) == 0 {
		t.Error("expected at least one reask rule from feedback recipe")
	}
	if len(cfg.Decisions) == 0 {
		t.Error("expected at least one decision from feedback recipe")
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	cfg2, errs2 := Compile(dslText)
	if len(errs2) > 0 {
		t.Fatalf("Round-trip compile errors: %v", errs2)
	}

	if len(cfg2.ReaskRules) != len(cfg.ReaskRules) {
		t.Errorf("round-trip reask rules: %d -> %d", len(cfg.ReaskRules), len(cfg2.ReaskRules))
	}
	if len(cfg2.Decisions) != len(cfg.Decisions) {
		t.Errorf("round-trip decisions: %d -> %d", len(cfg.Decisions), len(cfg2.Decisions))
	}
}

func TestMaintainedFeedbackRecipeValidates(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "feedback", "feedback-router.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("feedback recipe DSL not found: %v", err)
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
