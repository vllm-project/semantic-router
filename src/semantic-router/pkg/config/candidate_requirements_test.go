package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestRecipePoliciesRoundTripAndIsolation(t *testing.T) {
	input := []byte(`version: v0.3
routing:
  candidate_requirements: {capabilities: declared}
  data_policy: {replay: false}
recipes:
  - name: constrained
    routing:
      candidate_requirements: {context: known_limits}
      data_policy: {replay: true}
  - name: compatible
    routing: {}
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
	if scoped.CandidateRequirements.Capabilities != "" || scoped.CandidateRequirements.Context != CandidateContextKnownLimits || !scoped.DataPolicy.ReplayAllowed() {
		t.Fatal("recipe inherited another profile's policy")
	}
	if cfg.DataPolicy.ReplayAllowed() || cfg.CandidateRequirements.Context != "" {
		t.Fatal("default policy changed")
	}
	compatible, _ := cfg.RecipeByName("compatible")
	if compatible.Profile.CandidateRequirements != nil || compatible.Profile.DataPolicy != nil {
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
		if !reflect.DeepEqual(before.Profile.CandidateRequirements, after.Profile.CandidateRequirements) || !reflect.DeepEqual(before.Profile.DataPolicy, after.Profile.DataPolicy) {
			t.Fatalf("%s policy changed through canonical export", name)
		}
	}
	scoped.CandidateRequirements.Context = ""
	*scoped.DataPolicy.Replay = false
	if recipe.Profile.CandidateRequirements.Context != CandidateContextKnownLimits || !recipe.Profile.DataPolicy.ReplayAllowed() {
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
			input := "version: v0.3\n" + prefix + policy + "\n"
			if strings.HasPrefix(prefix, "recipes:") {
				input = "version: v0.3\nrouting: {}\n" + prefix + policy + "\n"
			}
			if _, err := ParseYAMLBytes([]byte(input)); err == nil {
				t.Errorf("accepted invalid policy: %s", input)
			}
		}
	}
	if _, err := ParseYAMLBytes([]byte("version: v0.3\nrouting:\n  data_policy: {replay: false}\nrecipes:\n  - name: default\n    routing: {}\n")); err == nil {
		t.Fatal("policy-only default scope did not conflict with explicit default recipe")
	}
}

func TestModelOutputLimitSurvivesCanonicalAndFragment(t *testing.T) {
	input := []byte(`version: v0.3
providers:
  models:
    - name: local
      backend_refs:
        - name: local
          endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: local
      capabilities: [chat]
      context_window_size: 8192
      max_output_tokens: 1024
  candidate_requirements: {context: known_limits}
`)
	cfg, err := ParseYAMLBytes(input)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelConfig["local"].MaxOutputTokens != 1024 {
		t.Fatal("effective catalog lost output limit")
	}
	fragment, err := yaml.Marshal(struct {
		Routing CanonicalRouting `yaml:"routing"`
	}{CanonicalRoutingFromRouterConfig(cfg)})
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
