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

	if !strings.Contains(dslText, "SIGNAL kb") {
		t.Error("decompiled DSL missing kb signal")
	}
	if !strings.Contains(dslText, "PLUGIN tools") {
		t.Error("decompiled DSL missing tools plugin directives")
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

	if len(cfg.KBRules) == 0 {
		t.Error("expected at least one kb rule from privacy recipe")
	}

	hasToolsPlugin := false
	for _, dec := range cfg.Decisions {
		if dec.GetToolsConfig() != nil {
			hasToolsPlugin = true
			break
		}
	}
	if !hasToolsPlugin {
		t.Error("expected at least one decision with tools plugin from privacy recipe")
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

	if len(cfg2.KBRules) != len(cfg.KBRules) {
		t.Errorf("round-trip kb rules: %d → %d",
			len(cfg.KBRules), len(cfg2.KBRules))
	}
	if len(cfg2.Decisions) != len(cfg.Decisions) {
		t.Errorf("round-trip Decisions: %d → %d",
			len(cfg.Decisions), len(cfg2.Decisions))
	}

	for i, dec := range cfg.Decisions {
		if i >= len(cfg2.Decisions) {
			continue
		}
		left := dec.GetToolsConfig()
		right := cfg2.Decisions[i].GetToolsConfig()
		switch {
		case (left == nil) != (right == nil):
			t.Errorf("round-trip decision %q tools plugin presence mismatch", dec.Name)
		case left != nil && right != nil && left.EffectiveMode() != right.EffectiveMode():
			t.Errorf("round-trip decision %q tools mode: %q → %q", dec.Name, left.EffectiveMode(), right.EffectiveMode())
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
