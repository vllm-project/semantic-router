package config

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v2"
)

func TestRecipesInheritGlobalDefaultsIndependently(t *testing.T) {
	for _, explicitDefault := range []bool{false, true} {
		t.Run(fmt.Sprintf("explicit_default=%t", explicitDefault), func(t *testing.T) {
			defaultProfile := `routing:
  strategy: priority
  fallback:
    enabled: false
    max_attempts: 2
    total_timeout: 12s
    per_attempt_timeout: 1s
recipes:
`
			if explicitDefault {
				defaultProfile = `recipes:
  - name: default
    routing:
      strategy: priority
      fallback:
        enabled: false
        max_attempts: 2
        total_timeout: 12s
        per_attempt_timeout: 1s
`
			}
			input := `version: v0.3
global:
  router:
    strategy: confidence
    fallback:
      enabled: true
      max_attempts: 5
      total_timeout: 50s
      per_attempt_timeout: 7s
` + defaultProfile + `  - name: inherited
    routing: {}
  - name: partial
    routing:
      fallback:
        max_attempts: 3
  - name: disabled
    routing:
      fallback:
        enabled: false
entrypoints:
  - model_names: [public/inherited]
    recipe: inherited
`
			cfg, err := ParseYAMLBytes([]byte(input))
			if err != nil {
				t.Fatal(err)
			}
			assertRecipe := func(name RecipeName, strategy RoutingStrategy, enabled bool, attempts int, timeout time.Duration) {
				t.Helper()
				recipe, ok := cfg.RecipeByName(name)
				if !ok {
					t.Fatalf("recipe %q missing", name)
				}
				scoped := cfg.ConfigForRecipe(recipe)
				if scoped.Strategy != strategy || scoped.Fallback == nil || scoped.Fallback.Enabled != enabled || scoped.Fallback.MaxAttempts != attempts || scoped.Fallback.TotalTimeout != timeout {
					t.Fatalf("recipe %q: strategy=%s fallback=%+v", name, scoped.Strategy, scoped.Fallback)
				}
			}
			assertRecipe(DefaultRecipeName, RoutingStrategyPriority, false, 2, 12*time.Second)
			assertRecipe("inherited", RoutingStrategyConfidence, true, 5, 50*time.Second)
			assertRecipe("partial", RoutingStrategyConfidence, true, 3, 50*time.Second)
			assertRecipe("disabled", RoutingStrategyConfidence, false, 5, 50*time.Second)

			selected, ok := cfg.RecipeForRoutingModel("public/inherited")
			if !ok || cfg.ConfigForRecipe(selected).Fallback.PerAttemptTimeout != 7*time.Second {
				t.Fatal("entrypoint did not select the independently resolved recipe")
			}

			// Exporting either the full runtime or a recipe view must preserve
			// global defaults, rather than publishing a recipe's overrides there.
			for _, snapshot := range []*RouterConfig{cfg, cfg.ConfigForRecipe(selected)} {
				global := CanonicalGlobalFromRouterConfig(snapshot)
				if global.Router.Strategy != RoutingStrategyConfidence || global.Router.Fallback == nil || global.Router.Fallback.MaxAttempts != 5 || !global.Router.Fallback.Enabled {
					t.Fatalf("export overwrote global defaults: %+v", global.Router)
				}
			}
			canonical := CanonicalConfigFromRouterConfig(cfg)
			canonical.Recipes = append(canonical.Recipes, CanonicalRecipe{Name: "added-after-export"})
			encoded, err := yaml.Marshal(canonical)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err = ParseYAMLBytes(encoded)
			if err != nil {
				t.Fatal(err)
			}
			assertRecipe("added-after-export", RoutingStrategyConfidence, true, 5, 50*time.Second)
			assertRecipe(DefaultRecipeName, RoutingStrategyPriority, false, 2, 12*time.Second)
		})
	}
}

func TestDefaultRecipeDoesNotSupplyMissingGlobalDefaults(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
routing:
  strategy: confidence
  fallback:
    enabled: true
    max_attempts: 8
recipes:
  - name: independent
    routing: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	recipe, _ := cfg.RecipeByName("independent")
	if recipe.Profile.Strategy != "" || recipe.Profile.Fallback != nil {
		t.Fatalf("default recipe leaked into named recipe: %+v", recipe.Profile)
	}
	global := CanonicalGlobalFromRouterConfig(cfg)
	if global.Router.Strategy != "" || global.Router.Fallback != nil {
		t.Fatalf("default recipe leaked into global export: %+v", global.Router)
	}
}

func TestRecipeFallbackValidationUsesGlobalNotDefaultRecipe(t *testing.T) {
	// This recipe is valid against the global 30-second limit, but invalid
	// against the unrelated default recipe's 3-second limit.
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
global:
  router:
    fallback:
      enabled: true
      total_timeout: 30s
      per_attempt_timeout: 2s
routing:
  fallback:
    total_timeout: 3s
recipes:
  - name: longer
    routing:
      fallback:
        per_attempt_timeout: 8s
`))
	if err != nil {
		t.Fatal(err)
	}
	recipe, _ := cfg.RecipeByName("longer")
	if recipe.Profile.Fallback.TotalTimeout != 30*time.Second || cfg.Fallback.TotalTimeout != 3*time.Second {
		t.Fatal("fallback validation and normalization disagree about the policy scope")
	}
	_, err = ParseYAMLBytes([]byte(`version: v0.3
global:
  router:
    strategy: invalid-global
routing:
  strategy: priority
`))
	if err == nil || !strings.Contains(err.Error(), "global.router.strategy") {
		t.Fatalf("local strategy must not hide an invalid global default: %v", err)
	}
}
