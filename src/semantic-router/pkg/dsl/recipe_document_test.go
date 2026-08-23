package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// mustOnlyRecipeConfig selects the model-free document owned by a maintained
// single-Recipe manifest. Full manifests intentionally keep Recipe state out
// of RouterConfig's flat routing view.
func mustOnlyRecipeConfig(t *testing.T, cfg *config.RouterConfig) *config.RouterConfig {
	t.Helper()
	if cfg == nil {
		t.Fatal("expected RouterConfig")
	}
	if len(cfg.Recipes) != 1 {
		t.Fatalf("expected exactly one Recipe, got %d", len(cfg.Recipes))
	}
	return cfg.ConfigForRecipe(&cfg.Recipes[0])
}
