package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const recipeTestBaseYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
    - name: model-b
      description: privacy tier
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent"]
  decisions:
    - name: default_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: model-a
          use_reasoning: false
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
    - name: model-b
      backend_refs:
        - endpoint: 127.0.0.1:8001
`

const recipeTestPrivacyBlockYAML = `
recipes:
  - name: privacy
    description: privacy profile
    routing:
      signals:
        keywords:
          - name: pii_keywords
            operator: OR
            keywords: ["ssn"]
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: pii_keywords
          modelRefs:
            - model: model-b
              use_reasoning: false
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
  - model_names: ["vllm-sr/default-alias"]
    recipe: default
`

func TestCanonicalRecipesRoutingOnlyNormalizesToDefaultRecipe(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if len(cfg.Recipes) != 1 {
		t.Fatalf("expected 1 normalized recipe, got %d", len(cfg.Recipes))
	}
	defaultRecipe := cfg.DefaultRecipe()
	if defaultRecipe == nil {
		t.Fatal("expected a default recipe")
	}
	if len(defaultRecipe.Decisions) != 1 || defaultRecipe.Decisions[0].Name != "default_route" {
		t.Fatalf("default recipe does not mirror the top-level routing profile: %+v", defaultRecipe.Decisions)
	}
	if len(cfg.Entrypoints) != 0 {
		t.Fatalf("expected no entrypoints, got %+v", cfg.Entrypoints)
	}
}

func TestCanonicalRecipesWithEntrypoints(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if len(cfg.Recipes) != 2 {
		t.Fatalf("expected 2 normalized recipes, got %d", len(cfg.Recipes))
	}

	privacy, ok := cfg.RecipeForRequestModel("vllm-sr/privacy")
	if !ok {
		t.Fatal("expected vllm-sr/privacy to resolve to a recipe")
	}
	if privacy.Name != "privacy" || len(privacy.Decisions) != 1 || privacy.Decisions[0].Name != "privacy_route" {
		t.Fatalf("unexpected privacy recipe: %+v", privacy)
	}
	if privacy.Decisions[0].ModelRefs[0].UseReasoning == nil {
		t.Fatal("expected recipe decisions to receive modelRef defaults")
	}
}

func TestCanonicalEntrypointsResolveDefaultRecipeAlias(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	viaAlias, ok := cfg.RecipeForRequestModel("vllm-sr/default-alias")
	if !ok || viaAlias.Name != DefaultRecipeName {
		t.Fatalf("expected vllm-sr/default-alias to resolve to the default recipe, got %+v", viaAlias)
	}
	if len(viaAlias.Decisions) != 1 || viaAlias.Decisions[0].Name != "default_route" {
		t.Fatalf("default recipe does not mirror the flat routing fields: %+v", viaAlias.Decisions)
	}

	if _, ok := cfg.RecipeForRequestModel("model-a"); ok {
		t.Fatal("expected a plain model name to miss the entrypoint table")
	}
}

func TestCanonicalRecipesOnlyDefaultBridgesFlatFields(t *testing.T) {
	yaml := `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
recipes:
  - name: default
    routing:
      signals:
        keywords:
          - name: urgent_keywords
            operator: OR
            keywords: ["urgent"]
      decisions:
        - name: recipe_default_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: urgent_keywords
          modelRefs:
            - model: model-a
              use_reasoning: false
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
`
	cfg, err := ParseYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if len(cfg.Decisions) != 1 || cfg.Decisions[0].Name != "recipe_default_route" {
		t.Fatalf("expected the explicit default recipe to bridge into the flat routing fields, got %+v", cfg.Decisions)
	}
	if len(cfg.Recipes) != 1 || cfg.Recipes[0].Name != DefaultRecipeName {
		t.Fatalf("expected a single default recipe, got %+v", cfg.Recipes)
	}
}

func TestCanonicalExportEmitsRecipesAndEntrypoints(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	canonical := CanonicalConfigFromRouterConfig(cfg)
	if len(canonical.Recipes) != 1 || canonical.Recipes[0].Name != "privacy" {
		t.Fatalf("expected the privacy recipe to be exported, got %+v", canonical.Recipes)
	}
	if len(canonical.Recipes[0].Routing.ModelCards) != 0 {
		t.Fatalf("exported recipes must not own model cards, got %+v", canonical.Recipes[0].Routing.ModelCards)
	}
	if len(canonical.Entrypoints) != 2 {
		t.Fatalf("expected 2 exported entrypoints, got %+v", canonical.Entrypoints)
	}

	static := CanonicalStaticConfigFromRouterConfig(cfg)
	if len(static.Recipes) != 0 || len(static.Entrypoints) != 0 {
		t.Fatalf("static export must not carry recipes or entrypoints, got %+v / %+v", static.Recipes, static.Entrypoints)
	}
}

func TestCanonicalExportRoundTripsRecipesAndEntrypoints(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + recipeTestPrivacyBlockYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("unexpected marshal error: %v", err)
	}
	reparsed, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("exported config failed to re-parse: %v", err)
	}

	if !reflect.DeepEqual(cfg.Recipes, reparsed.Recipes) {
		t.Fatalf("recipes did not round-trip:\nbefore: %+v\nafter: %+v", cfg.Recipes, reparsed.Recipes)
	}
	if !reflect.DeepEqual(cfg.Entrypoints, reparsed.Entrypoints) {
		t.Fatalf("entrypoints did not round-trip:\nbefore: %+v\nafter: %+v", cfg.Entrypoints, reparsed.Entrypoints)
	}
}

var canonicalRecipeErrorCases = []struct {
	name    string
	extra   string
	wantErr string
}{
	{
		name: "duplicate recipe name",
		extra: `
recipes:
  - name: privacy
    routing: {}
  - name: privacy
    routing: {}
`,
		wantErr: "duplicate recipe name",
	},
	{
		name: "default recipe conflicts with top-level routing",
		extra: `
recipes:
  - name: default
    routing: {}
`,
		wantErr: "conflicts with the top-level routing profile",
	},
	{
		name: "recipe owns model cards",
		extra: `
recipes:
  - name: privacy
    routing:
      modelCards:
        - name: rogue-model
`,
		wantErr: "the model catalog is shared",
	},
	{
		name: "recipe decision references unknown model",
		extra: `
recipes:
  - name: privacy
    routing:
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: urgent_keywords
          modelRefs:
            - model: missing-model
              use_reasoning: false
`,
		wantErr: "references unknown model",
	},
	{
		name: "entrypoint references unknown recipe",
		extra: `
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: missing
`,
		wantErr: "unknown recipe",
	},
	{
		name: "entrypoint without model names",
		extra: `
entrypoints:
  - model_names: []
    recipe: default
`,
		wantErr: "model_names cannot be empty",
	},
	{
		name: "duplicate model name across entrypoints",
		extra: `
recipes:
  - name: privacy
    routing: {}
entrypoints:
  - model_names: ["vllm-sr/dup"]
    recipe: default
  - model_names: ["vllm-sr/dup"]
    recipe: privacy
`,
		wantErr: "already mapped by another entrypoint",
	},
}

func TestCanonicalRecipeValidationErrors(t *testing.T) {
	for _, testCase := range canonicalRecipeErrorCases {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(recipeTestBaseYAML + testCase.extra))
			if err == nil {
				t.Fatalf("expected parse error containing %q", testCase.wantErr)
			}
			if !strings.Contains(err.Error(), testCase.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", testCase.wantErr, err)
			}
		})
	}
}
