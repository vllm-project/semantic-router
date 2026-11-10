package dsl

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedFeedbackRecipeParsesAndDecompilesWithoutError(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "feedback", "config.yaml")
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

func TestMaintainedFeedbackRecipeValidates(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "config", "recipes", "feedback", "recipe.dsl")
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
