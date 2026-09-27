package dsl

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRecipePrototypeScoringSurvivesPublicationRoundTrip(t *testing.T) {
	disabled := false
	overrides := []*config.PrototypeScoringConfig{
		nil,
		{},
		{Enabled: &disabled, BestWeight: 0.75, TopM: 2},
		{ClusterSimilarityThreshold: 0.8, MaxPrototypes: 12, BestWeight: 1, TopM: 3, MarginThreshold: 0.05},
	}
	for _, override := range overrides {
		cfg := &config.RouterConfig{}
		for _, name := range []config.RecipeName{"inherited", "authored"} {
			var ruleOverride *config.PrototypeScoringConfig
			if name == "authored" {
				ruleOverride = override
			}
			cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{
				Name: name,
				Profile: config.RoutingProfile{Signals: config.Signals{
					EmbeddingRules:  []config.EmbeddingRule{{Name: "intent", SimilarityThreshold: 0.8, Candidates: []string{"sample"}, PrototypeScoring: ruleOverride}},
					ComplexityRules: []config.ComplexityRule{{Name: "difficulty", Threshold: 0.1, Hard: config.ComplexityCandidates{Candidates: []string{"hard"}, ImageCandidates: []string{"hard.png"}}, Easy: config.ComplexityCandidates{Candidates: []string{"easy"}}, PrototypeScoring: ruleOverride}},
				}},
			})
		}
		// Canonical publication precedes both textual DSL and AST consumers.
		published, err := yaml.Marshal(config.CanonicalConfigFromRouterConfig(cfg))
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := config.ParseYAMLBytes(published)
		if err != nil {
			t.Fatal(err)
		}
		source, err := DecompileConfig(parsed)
		if err != nil {
			t.Fatal(err)
		}
		textual, errs := Compile(source)
		if len(errs) != 0 {
			t.Fatalf("compile: %v", errs)
		}
		ast, errs := CompileAST(DecompileToAST(parsed))
		if len(errs) != 0 {
			t.Fatalf("compile AST: %v", errs)
		}
		for _, restored := range []*config.RouterConfig{parsed, textual, ast} {
			for _, name := range []config.RecipeName{"inherited", "authored"} {
				recipe, ok := restored.RecipeByName(name)
				if !ok {
					t.Fatalf("lost recipe %s", name)
				}
				scoped := restored.ConfigForRecipe(recipe)
				var want *config.PrototypeScoringConfig
				if name == "authored" {
					want = override
				}
				for _, got := range []*config.PrototypeScoringConfig{scoped.EmbeddingRules[0].PrototypeScoring, scoped.ComplexityRules[0].PrototypeScoring} {
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("recipe %s override = %+v, want %+v", name, got, want)
					}
				}
			}
		}
	}
}

func TestPrototypeScoringDSLRejectsMalformedOverrides(t *testing.T) {
	for _, value := range []string{`false`, `{ enabled: "nope" }`, `{ unknown_field: 10 }`} {
		for _, signal := range []string{"embedding", "complexity"} {
			source := "SIGNAL " + signal + " sample { prototype_scoring: " + value + " }"
			if _, errs := Compile(source); len(errs) == 0 {
				t.Fatalf("accepted malformed override: %s", source)
			}
			diagnostics, _ := Validate(source)
			found := false
			for _, diagnostic := range diagnostics {
				found = found || strings.Contains(diagnostic.Message, "prototype_scoring")
			}
			if !found {
				t.Fatalf("validator lost override error: %s", source)
			}
		}
	}
}
