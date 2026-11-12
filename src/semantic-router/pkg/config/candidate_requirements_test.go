package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestRecipePoliciesRoundTripAndIsolation(t *testing.T) {
	input := []byte(`version: v0.5
recipes:
  - name: default
    candidate_requirements: {capabilities: declared}
  - name: constrained
    candidate_requirements: {context: known_limits}
  - name: compatible
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}
	recipe, ok := cfg.RecipeByName("constrained")
	if !ok {
		t.Fatal("missing constrained recipe")
	}
	scoped := cfg.ConfigForRecipe(recipe)
	if scoped.CandidateRequirements.Capabilities != "" || scoped.CandidateRequirements.Context != CandidateContextKnownLimits {
		t.Fatal("recipe inherited another profile's policy")
	}
	if cfg.CandidateRequirements.Context != "" {
		t.Fatal("default policy changed")
	}
	compatible, _ := cfg.RecipeByName("compatible")
	if compatible.Profile.CandidateRequirements != nil {
		t.Fatal("absent policy inherited")
	}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	again, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []RecipeName{DefaultRecipeName, "constrained", "compatible"} {
		before, _ := cfg.RecipeByName(name)
		after, _ := again.RecipeByName(name)
		if !reflect.DeepEqual(before.Profile.CandidateRequirements, after.Profile.CandidateRequirements) {
			t.Fatalf("%s policy changed through canonical export", name)
		}
	}
	scoped.CandidateRequirements.Context = ""
	if recipe.Profile.CandidateRequirements.Context != CandidateContextKnownLimits {
		t.Fatal("scoped mutation aliased recipe")
	}
}

func TestCandidateRequirementsRejectInvalidCanonicalPolicies(t *testing.T) {
	for _, policy := range []string{
		"candidate_requirements: {capabilities: inferred}",
		"candidate_requirements: {context: bounded}",
		"candidate_requirements: {context: known_limits, minimum: 2}",
		"data_policy: {replay: false, export: true}",
	} {
		if _, err := ParseRoutingYAMLBytes([]byte("routing:\n  " + policy + "\n")); err == nil {
			t.Errorf("routing fragment accepted invalid policy: %s", policy)
		}
		for _, prefix := range []string{"routing:\n  ", "recipes:\n  - name: isolated\n    routing:\n      "} {
			input := "version: v0.5\n" + prefix + policy + "\n"
			if strings.HasPrefix(prefix, "recipes:") {
				input = "version: v0.5\nrouting: {}\n" + prefix + policy + "\n"
			}
			if _, err := ParseYAMLBytes([]byte(input)); err == nil {
				t.Errorf("accepted invalid policy: %s", input)
			}
		}
	}
}

func TestModelOutputLimitSurvivesCanonicalAndFragment(t *testing.T) {
	input := []byte(`version: v0.5
providers:
  models:
    - name: local
      backend_refs:
        - name: local
          endpoint: 127.0.0.1:8000
      capabilities: [chat]
      context_window_size: 8192
      max_output_tokens: 1024
recipes:
  - name: default
    candidate_requirements: {context: known_limits}
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelConfig["local"].MaxOutputTokens != 1024 {
		t.Fatal("effective catalog lost output limit")
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	facts := CanonicalProviderModel{Name: "local", ModelFacts: canonical.Providers.Models[0].ModelFacts}
	fragment, err := yaml.Marshal(CanonicalConfig{
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{facts}},
		Recipes:   canonical.Recipes,
	})
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseRoutingYAMLBytes(fragment)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ModelConfig["local"].MaxOutputTokens != 1024 || parsed.CandidateRequirements.Context != CandidateContextKnownLimits {
		t.Fatal("routing fragment lost output limit or policy")
	}
}
