package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRecipeClassifiersDoNotOwnGlobalCacheNLI(t *testing.T) {
	for _, scope := range []config.RecipeName{config.DefaultRecipeName, "first", "second"} {
		cfg := &config.RouterConfig{RoutingScope: scope}
		cfg.SemanticCache.Enabled = true
		cfg.SemanticCache.PolarityGuard = &config.PolarityGuardConfig{Mode: "nli"}
		cfg.HallucinationMitigation.NLIModel.ModelID = "/not-installed/unused-recipe-override"
		c := &Classifier{Config: cfg}
		for _, task := range append(c.runtimeTasks(), c.defaultAPIRuntimeTasks()...) {
			if task.Name == "classifier.semantic_cache_nli" || task.Name == "classifier.hallucination" {
				t.Fatalf("recipe %s prepared a service-owned NLI: %s", scope, task.Name)
			}
		}
	}
}
