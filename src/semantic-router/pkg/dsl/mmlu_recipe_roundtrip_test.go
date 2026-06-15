package dsl

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedMMLURecipeParsesAndDecompilesWithoutError(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "mmlu", "mmlu-router.yaml")
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

	if len(cfg.KnowledgeBases) == 0 {
		t.Fatal("expected built-in knowledge bases after parsing mmlu recipe")
	}

	found := false
	for _, kb := range cfg.KnowledgeBases {
		if kb.Name == "mmlu_kb" {
			found = true
			if kb.Source.Path != "knowledge_bases/mmlu/" {
				t.Fatalf("mmlu_kb source path = %q", kb.Source.Path)
			}
			break
		}
	}
	if !found {
		t.Fatal("expected mmlu recipe to resolve built-in mmlu_kb default")
	}

	if len(cfg.KBRules) == 0 {
		t.Fatal("expected mmlu recipe to define kb signals")
	}
}

func TestMaintainedMMLURecipeDSLRoundTrip(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "mmlu", "mmlu-router.dsl")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Skipf("mmlu recipe DSL not found at %s: %v", assetPath, err)
	}

	cfg, errs := Compile(string(data))
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}

	if len(cfg.KBRules) == 0 {
		t.Error("expected at least one kb rule from mmlu recipe")
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

	if len(cfg2.KBRules) != len(cfg.KBRules) {
		t.Errorf("round-trip kb rules: %d -> %d", len(cfg.KBRules), len(cfg2.KBRules))
	}
	if len(cfg2.Decisions) != len(cfg.Decisions) {
		t.Errorf("round-trip decisions: %d -> %d", len(cfg.Decisions), len(cfg2.Decisions))
	}
}

func TestMaintainedMMLURecipeValidates(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "recipes", "mmlu", "mmlu-router.dsl")
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
