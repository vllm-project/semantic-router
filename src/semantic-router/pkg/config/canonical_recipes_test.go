package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

const recipeTestBaseYAML = `
version: v0.4
models:
  - name: model-a
    card:
      description: default tier
      loras: [general-expert]
    connections:
      - provider: private-test
        endpoint: http://127.0.0.1:8000
        model: model-a
  - name: model-b
    card: {description: privacy tier}
    connections:
      - provider: private-test
        endpoint: http://127.0.0.1:8001
        model: model-b
recipes:
  - name: default
    document:
      signals:
        keywords:
          - name: urgent_keywords
            operator: OR
            keywords: [urgent]
      decisions:
        - name: default_route
          rules:
            operator: AND
            conditions:
              - {type: keyword, name: urgent_keywords}
entrypoints:
  - name: vllm-sr/default
    aliases: [vllm-sr/default-alias]
    recipe: default
    assignments:
      default_route:
        models: [{model: model-a}]
global:
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
`

const recipeTestPrivacyYAML = `
version: v0.4
models:
  - name: model-a
    card: {description: default tier}
    connections:
      - {provider: private-test, endpoint: http://127.0.0.1:8000, model: model-a}
  - name: model-b
    card: {description: privacy tier}
    connections:
      - {provider: private-test, endpoint: http://127.0.0.1:8001, model: model-b}
recipes:
  - name: default
    document:
      signals:
        keywords:
          - {name: urgent_keywords, operator: OR, keywords: [urgent]}
      decisions:
        - name: default_route
          rules:
            operator: AND
            conditions: [{type: keyword, name: urgent_keywords}]
  - name: privacy
    description: privacy profile
    document:
      signals:
        keywords:
          - {name: pii_keywords, operator: OR, keywords: [ssn]}
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions: [{type: keyword, name: pii_keywords}]
entrypoints:
  - name: vllm-sr/default
    aliases: [vllm-sr/default-alias]
    recipe: default
    assignments:
      default_route:
        models: [{model: model-a}]
  - name: vllm-sr/privacy
    recipe: privacy
    assignments:
      privacy_route:
        models: [{model: model-b}]
global:
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
`

func parseRecipeFixtureYAML(t *testing.T, input []byte) (*RouterConfig, error) {
	t.Helper()
	return testAuthoringParser(t).ParseYAMLBytes(input)
}

func TestCanonicalRecipeCompilesOnlyExplicitResources(t *testing.T) {
	cfg, err := parseRecipeFixtureYAML(t, []byte(recipeTestBaseYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if len(cfg.Recipes) != 1 || cfg.Recipes[0].Name != DefaultRecipeName {
		t.Fatalf("compiled Recipes = %+v", cfg.Recipes)
	}
	if len(cfg.Entrypoints) != 1 || cfg.Entrypoints[0].Name != "vllm-sr/default" {
		t.Fatalf("compiled Entrypoints = %+v", cfg.Entrypoints)
	}
	if _, ok := cfg.RecipeForRequestModel("vllm-sr/default-alias"); !ok {
		t.Fatal("explicit Entrypoint alias did not resolve")
	}
	if _, ok := cfg.RecipeForRequestModel("model-a"); ok {
		t.Fatal("physical Model name became an implicit Entrypoint")
	}
}

func TestCanonicalExportDoesNotInventSourceModelsWithoutSnapshot(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.ModelConfig = map[string]ModelParams{"orphan": {Description: "runtime-only"}}
	canonical := CanonicalConfigFromRouterConfig(&cfg)
	if len(canonical.Models) != 0 {
		t.Fatalf("canonical export invented incomplete source Models: %+v", canonical.Models)
	}
}

func TestCanonicalRecipesRemainIsolated(t *testing.T) {
	cfg, err := parseRecipeFixtureYAML(t, []byte(recipeTestPrivacyYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	privacy, ok := cfg.RecipeForRequestModel("vllm-sr/privacy")
	if !ok || privacy.Name != "privacy" || len(privacy.Profile.Decisions) != 1 {
		t.Fatalf("privacy Recipe = %+v", privacy)
	}
	if privacy.Profile.Decisions[0].ModelRefs[0].Model != "model-b" {
		t.Fatalf("privacy assignments = %+v", privacy.Profile.Decisions[0].ModelRefs)
	}
	defaultRecipe, ok := cfg.RecipeForRequestModel("vllm-sr/default")
	if !ok || defaultRecipe == nil || defaultRecipe.Profile.Decisions[0].ModelRefs[0].Model != "model-a" {
		t.Fatalf("default Recipe = %+v", defaultRecipe)
	}
}

func TestCanonicalExportEmitsHumanRecipesAndEntrypoints(t *testing.T) {
	cfg, err := parseRecipeFixtureYAML(t, []byte(recipeTestPrivacyYAML))
	if err != nil {
		t.Fatal(err)
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	if len(canonical.Models) != 2 || len(canonical.Recipes) != 2 || len(canonical.Entrypoints) != 2 {
		t.Fatalf("canonical export = %+v", canonical)
	}
	for _, recipe := range canonical.Recipes {
		for _, decision := range recipe.Document.Decisions {
			if decision.ID != "" || len(decision.ModelRefs) != 0 {
				t.Fatalf("Recipe export contains compiled state: %+v", decision)
			}
		}
	}
	routingExport, err := yaml.Marshal(struct {
		Models      []AuthoringModel      `yaml:"models"`
		Recipes     []AuthoringRecipe     `yaml:"recipes"`
		Entrypoints []AuthoringEntrypoint `yaml:"entrypoints"`
	}{canonical.Models, canonical.Recipes, canonical.Entrypoints})
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"provider_catalog_revision:", "backends:", "model_id:", "recipe_id:", "revision:"} {
		if strings.Contains(string(routingExport), forbidden) {
			t.Fatalf("human routing export contains %q:\n%s", forbidden, routingExport)
		}
	}
	exported, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	reparsed, err := testAuthoringParser(t).ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("exported config failed to reparse: %v", err)
	}
	if !reflect.DeepEqual(cfg.Recipes, reparsed.Recipes) || !reflect.DeepEqual(cfg.Entrypoints, reparsed.Entrypoints) {
		t.Fatalf("routing resources did not round trip")
	}
}

func TestCanonicalRecipeValidationErrors(t *testing.T) {
	tests := []struct {
		name, needle, replacement, want string
	}{
		{"duplicate Recipe", "  - name: privacy\n", "  - name: default\n", "duplicate recipe name"},
		{"unknown Recipe", "    recipe: privacy\n", "    recipe: missing\n", "unknown Recipe"},
		{"unknown Model", "        models: [{model: model-b}]", "        models: [{model: missing}]", "unknown Model"},
		{"duplicate alias", "  - name: vllm-sr/privacy\n", "  - name: vllm-sr/privacy\n    aliases: [vllm-sr/default-alias]\n", "already mapped by another entrypoint"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			document := strings.Replace(recipeTestPrivacyYAML, test.needle, test.replacement, 1)
			_, err := parseRecipeFixtureYAML(t, []byte(document))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestCanonicalRecipeNameRejectsSurroundingWhitespace(t *testing.T) {
	document := strings.Replace(recipeTestBaseYAML, "  - name: default\n", "  - name: ' default '\n", 1)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil || !strings.Contains(err.Error(), "surrounding whitespace") {
		t.Fatalf("error = %v", err)
	}
}

func TestDuplicateIdenticalSignalWithinRecipeRejected(t *testing.T) {
	needle := "          - {name: urgent_keywords, operator: OR, keywords: [urgent]}"
	document := strings.Replace(recipeTestPrivacyYAML, needle, needle+"\n"+needle, 1)
	_, err := parseRecipeFixtureYAML(t, []byte(document))
	if err == nil || !strings.Contains(err.Error(), "duplicate local name") {
		t.Fatalf("error = %v", err)
	}
}
