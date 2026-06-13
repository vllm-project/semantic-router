package config

import (
	"testing"

	"gopkg.in/yaml.v2"
)

func TestCanonicalServiceGlobalSessionTokenBudgetRoundTrips(t *testing.T) {
	const yamlInput = `
router: {}
services:
  session_token_budget:
    enabled: true
    budget_tokens: 40000
    thresholds:
      shape_tools: 1.0
      compress: 1.5
      downgrade: 2.0
      terminate: 3.0
stores: {}
integrations: {}
model_catalog:
  embeddings: {}
  system: {}
  modules: {}
`
	var global CanonicalGlobal
	if err := yaml.Unmarshal([]byte(yamlInput), &global); err != nil {
		t.Fatalf("failed to unmarshal canonical global: %v", err)
	}
	if !global.Services.SessionTokenBudget.Enabled || global.Services.SessionTokenBudget.BudgetTokens != 40000 {
		t.Fatalf("expected canonical global to surface session_token_budget, got %+v", global.Services.SessionTokenBudget)
	}

	cfg := &RouterConfig{}
	if err := applyCanonicalGlobal(cfg, &global); err != nil {
		t.Fatalf("applyCanonicalGlobal failed: %v", err)
	}
	if !cfg.SessionTokenBudget.Enabled || cfg.SessionTokenBudget.BudgetTokens != 40000 {
		t.Fatalf("applyCanonicalGlobal should propagate session_token_budget, got %+v", cfg.SessionTokenBudget)
	}
	if cfg.SessionTokenBudget.Thresholds.Terminate != 3.0 {
		t.Fatalf("expected terminate threshold 3.0, got %v", cfg.SessionTokenBudget.Thresholds.Terminate)
	}

	exported := CanonicalGlobalFromRouterConfig(cfg)
	if exported == nil {
		t.Fatal("expected non-nil canonical global on export")
	}
	if exported.Services.SessionTokenBudget.BudgetTokens != 40000 {
		t.Fatalf("CanonicalGlobalFromRouterConfig should round-trip session_token_budget, got %+v", exported.Services.SessionTokenBudget)
	}
}
