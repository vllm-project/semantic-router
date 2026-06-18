package dsl

import "testing"

func TestCompileASTSeedsModelSelectionDefaults(t *testing.T) {
	cfg, errs := CompileAST(&Program{})
	if len(errs) > 0 {
		t.Fatalf("CompileAST errors: %v", errs)
	}
	if cfg == nil {
		t.Fatal("expected router config from CompileAST")
	}
	if !cfg.ModelSelection.RouterDC.UseQueryContrastive ||
		!cfg.ModelSelection.RouterDC.UseModelContrastive ||
		!cfg.ModelSelection.RouterDC.UseCapabilities {
		t.Fatalf("expected CompileAST to seed RouterDC defaults, got %+v", cfg.ModelSelection.RouterDC)
	}
	if !cfg.ModelSelection.Hybrid.NormalizeScores {
		t.Fatalf("expected CompileAST to seed Hybrid defaults, got %+v", cfg.ModelSelection.Hybrid)
	}
}
